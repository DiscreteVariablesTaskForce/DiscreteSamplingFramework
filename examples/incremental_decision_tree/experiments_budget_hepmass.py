"""
HEPMASS at the budget of experiments_budget.py: a second tall dataset beside
susy, with 8.4 million training rows of 27 continuous features and two
balanced classes. It is a run of its own because splitting it needs about
12 GB, more than the one core the prepare job has by default, and because its
2.1 million test rows make it the slowest to evaluate:

    RUN_ID=ch3_budget_hepmass EXPERIMENTS=examples/incremental_decision_tree/experiments_budget_hepmass.py \\
        PREP_CORES=4 CORES=64 TIME=8:00:00 EVAL_CORES=64 EVAL_TIME=24:00:00 slurm/submit.sh
"""
from experiments_budget import experiments

EXPERIMENTS = experiments(("hepmass",))
