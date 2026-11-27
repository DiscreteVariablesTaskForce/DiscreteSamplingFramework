"""
Large-tree runs: the equal-budget comparison of experiments_budget_marginal.py
(marginal-likelihood split proposal, cheaper HINTS bookkeeping) with the tree-size
cap raised from 100 to a value chosen by a pilot run, so that the cap does not bind.
With 100 nodes, every SUSY chain reaches the cap within a few hundred passes and
spends the rest of its budget there, which hides the differences between samplers.

Two stages, both driven by slurm/bigtree_pipeline.sh:

  GROUPS["pilot"]  SUSY and HEPMASS, which grow the largest trees: MH, and DA and
                   HINTS at 1/8 and at their smallest block size, PILOT_CHAINS
                   chains each, the full budget, a safety cap of PILOT_CAP nodes.
  decide_tree_cap.py reads the largest tree the pilot reached and writes the cap
                   (SAFETY times it, rounded up, at least FLOOR) to CAP_FILE.
  GROUPS["mcmc"], ["mcmc_hepmass"], ["smc"], ["smc_hepmass"]
                   the lists of experiments_budget_marginal.py with that cap, and
                   fewer stored records (MCMC_RECORDS, SMC_RECORDS) to keep the
                   stored trees affordable at a few thousand nodes.

The main groups read CAP_FILE when they are built. Before the pilot has decided,
listing them on the login node (submit.sh's --count/--list) shows no cap; inside a
Slurm job a missing CAP_FILE is an error, so no run can start without the decision.
"""
import json
import math
import os
import sys

import experiments_budget as mcmc_lists
import experiments_budget_smc as smc_lists

PILOT_ID = "bigtree_pilot"
CAP_FILE = os.environ.get("DSF_TREE_CAP_FILE", os.path.join("Results", PILOT_ID, "tree_cap.json"))
PILOT_CAP = 20_000          # safety cap for the pilot only
PILOT_CHAINS = 4
PILOT_RECORDS = 200
MCMC_RECORDS = 500
SMC_RECORDS = 20
CAP_X = 10                  # iteration/step caps: upper limits only, runs stop at their budget
SPLITS = dict(split_proposal="marginal", threshold_proposal="data")


def tree_cap():
    """The cap the pilot decided, or None before it has (login node only)."""
    if os.path.exists(CAP_FILE):
        with open(CAP_FILE) as f:
            return int(json.load(f)["cap"])
    if os.environ.get("SLURM_JOB_ID"):
        raise SystemExit(f"{CAP_FILE} is missing: the pilot's decision job has not written the tree cap")
    print(f"[experiments_bigtree] {CAP_FILE} not written yet: the cap is set by the pilot",
          file=sys.stderr)
    return None


def adapt(entries, cap_key, records, cap):
    out = []
    for e in entries:
        e = dict(e, **SPLITS, records=records, max_tree_size=cap)
        if e["proposals"] == ["HINTS"]:
            e["fused_buckets"] = True
        e[cap_key] = CAP_X * e[cap_key]
        out.append(e)
    return out


def pilot():
    out = []
    for dataset in ("susy", "hepmass"):
        smallest = min(mcmc_lists.SS_PROPS[dataset])
        picks = [("MH", None)] + [(p, s) for p in ("DA", "HINTS") for s in (1 / 8, smallest)]
        for proposal, ss_prop in picks:
            knobs = (dict(levels=int(round(-math.log2(ss_prop))), branching=2)
                     if proposal == "HINTS" else {})
            e = mcmc_lists.entry(dataset, proposal, ss_prop, **knobs)
            e["chains"] = PILOT_CHAINS
            out.append(e)
    return adapt(out, "iters", PILOT_RECORDS, PILOT_CAP)


class LazyGroups(dict):
    """GROUPS whose lists are built when asked for, so that the pilot can be
    listed and run before the cap it decides exists."""

    def __getitem__(self, key):
        return dict.__getitem__(self, key)()


GROUPS = LazyGroups(
    pilot=pilot,
    mcmc=lambda: adapt(mcmc_lists.experiments(("susy", "digits", "sensorless")), "iters", MCMC_RECORDS, tree_cap()),
    mcmc_hepmass=lambda: adapt(mcmc_lists.experiments(("hepmass",)), "iters", MCMC_RECORDS, tree_cap()),
    smc=lambda: adapt(smc_lists.experiments(("susy", "digits", "sensorless")), "steps", SMC_RECORDS, tree_cap()),
    smc_hepmass=lambda: adapt(smc_lists.experiments(("hepmass",)), "steps", SMC_RECORDS, tree_cap()),
)

EXPERIMENTS = GROUPS["pilot"]
