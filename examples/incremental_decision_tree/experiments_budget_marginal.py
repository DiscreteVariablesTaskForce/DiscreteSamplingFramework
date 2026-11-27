"""
The equal-budget runs of experiments_budget.py (MCMC) and experiments_budget_smc.py
(SMC) again, with the marginal-likelihood split proposal: Lakshminarayanan, Roy and
Teh's (2013) optimal proposal computed on a random subset of 64 of the moved node's
rows, with a quarter of the uniform choice (split_proposal="marginal", see
moves._marginal_splits). Everything else is unchanged, so each run pairs, chain for
chain and start for start, with its uniform-split counterpart; HINTS uses the cheaper
block bookkeeping (fused_buckets) of experiments_budget_fused.py.

Informed splits change the row reads per iteration (MH and DA read fewer, since a
good proposal routes fewer rows), so the iteration and step caps are CAP_X times the
uniform runs' ones. They are only upper limits: every run stops at its budget.

One group per run (the file names do not carry the split proposal):

    E=examples/incremental_decision_tree/experiments_budget_marginal.py
    RUN_ID=ch3_budget_marginal EXPERIMENTS="$E::mcmc" \\
        SPLIT=4 CORES=16 TIME=12:00:00 EVAL_CORES=64 EVAL_TIME=24:00:00 slurm/submit.sh
    RUN_ID=ch3_budget_marginal_hepmass EXPERIMENTS="$E::mcmc_hepmass" \\
        PREP_CORES=4 SPLIT=4 CORES=16 TIME=24:00:00 EVAL_CORES=64 EVAL_TIME=24:00:00 slurm/submit.sh
    RUN_ID=ch4_smc_budget_marginal EXPERIMENTS="$E::smc" \\
        CORES=16 TIME=24:00:00 EVAL_CORES=32 slurm/submit.sh
    RUN_ID=ch4_smc_budget_marginal_hepmass EXPERIMENTS="$E::smc_hepmass" \\
        PREP_CORES=4 CORES=32 TIME=24:00:00 EVAL_CORES=32 EVAL_TIME=24:00:00 slurm/submit.sh
"""
import experiments_budget as mcmc_lists
import experiments_budget_smc as smc_lists

CAP_X = 10
SPLITS = dict(split_proposal="marginal", threshold_proposal="data")


def marginal(entries, cap_key):
    """The entries with the marginal-likelihood split proposal, HINTS with the
    cheaper bookkeeping, and CAP_X times the iteration (or step) cap."""
    out = []
    for e in entries:
        e = dict(e, **SPLITS)
        if e["proposals"] == ["HINTS"]:
            e["fused_buckets"] = True
        e[cap_key] = CAP_X * e[cap_key]
        out.append(e)
    return out


GROUPS = {
    "mcmc": marginal(mcmc_lists.experiments(("susy", "digits", "sensorless")), "iters"),
    "mcmc_hepmass": marginal(mcmc_lists.experiments(("hepmass",)), "iters"),
    "smc": marginal(smc_lists.experiments(("susy", "digits", "sensorless")), "steps"),
    "smc_hepmass": marginal(smc_lists.experiments(("hepmass",)), "steps"),
}

EXPERIMENTS = GROUPS["mcmc"]
