"""
The two-level HINTS hierarchy (2 levels x 4) on susy at ss_prop 1/16, at the
budget of experiments_budget.py. The file names carry ss_prop but not the shape
of the hierarchy, so this goes in a run of its own:

    RUN_ID=ch3_budget_2x4 EXPERIMENTS=examples/incremental_decision_tree/experiments_budget_shapes.py \\
        CORES=64 TIME=6:00:00 EVAL_CORES=64 slurm/submit.sh
"""
from experiments_budget import entry

EXPERIMENTS = [entry("susy", "HINTS", 1 / 16, key="HINTS2x4", levels=2, branching=4)]
