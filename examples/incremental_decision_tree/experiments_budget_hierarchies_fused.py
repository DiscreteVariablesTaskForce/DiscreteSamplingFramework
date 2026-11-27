"""
Every HINTS run of the hierarchy study, and the binary HINTS on HEPMASS, again
with fused_buckets (which implies lazy_counts): the same chains, seed for seed
and tree for tree, with the sweep's block bookkeeping at about a quarter of the
row reads (see proposals/hints.py and Results/overnight/REPORT.md).

The sources are:

    experiments_budget_hierarchies.py        split64 (HINTS only), one_level,
                                             branch4, branch8
    experiments_budget_mixed_hierarchies.py  1-2-4-8, 1-8-4-2, 2-4-8, 4-2-8,
                                             8-2-4, 8-4-2
    experiments_budget.py                    the binary HINTS on hepmass
                                             (hepmass_binary)

DA and Flat HINTS are not rerun, since nothing about them changes, and neither
is MH. The file names carry ss_prop but not the shape of the hierarchy, so each
group is a run of its own, read as GROUPS[name] through `<this file>::<name>`:

    E=examples/incremental_decision_tree/experiments_budget_hierarchies_fused.py
    for g in split64 one_level branch4 branch8 1-2-4-8 1-8-4-2 2-4-8 4-2-8 8-2-4 8-4-2 hepmass_binary; do
        PREP_CORES=4 SPLIT=4 CORES=16 TIME=24:00:00 EVAL_CORES=64 EVAL_TIME=24:00:00 \\
        RUN_ID=fused_$g EXPERIMENTS="$E::$g" slurm/submit.sh
    done

A chain runs about four times as many iterations within the same budget, at
about the same time per iteration, so it takes three to four times as long as
before: hence the longer TIME. The iteration caps of the source lists already
leave room for that (they assumed 5e5 row reads per iteration; the fused
HINTS reads about 1e6 on SUSY at 1/64), and on hepmass_binary they are raised
five-fold as in experiments_budget_fused.py.
"""
from experiments_budget import experiments as budget_experiments, iteration_cap
from experiments_budget_hierarchies import GROUPS as SHAPE_GROUPS
from experiments_budget_mixed_hierarchies import GROUPS as MIXED_GROUPS

FUSED_CAP = 5


def fused(entries):
    """The HINTS entries of a list, with fused_buckets set."""
    return [dict(e, fused_buckets=True) for e in entries if e["proposals"] == ["HINTS"]]


GROUPS = {name: fused(entries) for name, entries in {**SHAPE_GROUPS, **MIXED_GROUPS}.items()}

GROUPS["hepmass_binary"] = []
for e in fused(budget_experiments(("hepmass",))):
    e["iters"] = FUSED_CAP * iteration_cap("hepmass", "HINTS", e["ss_prop"])
    GROUPS["hepmass_binary"].append(e)

EXPERIMENTS = GROUPS["split64"]
