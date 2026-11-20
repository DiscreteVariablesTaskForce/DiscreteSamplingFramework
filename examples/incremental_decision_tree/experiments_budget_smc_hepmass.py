"""
HEPMASS SMC at the budget of experiments_budget_smc.py, in a run of its own.
An SMC run holds about twice its particle count of trees at once, and on
hepmass each tree carries the indices of 8.4 million rows, so a run needs more
memory than the one core per run the other SMC runs get; CORES=32 gives the 16
runs two cores' memory each:

    RUN_ID=ch4_smc_budget_hepmass EXPERIMENTS=examples/incremental_decision_tree/experiments_budget_smc_hepmass.py \\
        PREP_CORES=4 CORES=32 TIME=8:00:00 EVAL_CORES=32 EVAL_TIME=24:00:00 slurm/submit.sh
"""
from experiments_budget_smc import experiments

EXPERIMENTS = experiments(("hepmass",))
