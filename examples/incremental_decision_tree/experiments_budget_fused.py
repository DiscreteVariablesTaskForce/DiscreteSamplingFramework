"""
The HINTS runs of experiments_budget.py again, with fused_buckets (which implies
lazy_counts): the same chains, seed for seed and tree for tree, but with the
sweep's block bookkeeping done in about a fifth of the row reads (see
proposals/hints.py and Results/overnight/REPORT.md). A run of its own because
the file names do not carry the option; MH, DA and FlatHINTS are not rerun, since
nothing about them changes.

    RUN_ID=ch3_budget_fused EXPERIMENTS=examples/incremental_decision_tree/experiments_budget_fused.py \\
        CORES=64 TIME=8:00:00 EVAL_CORES=64 slurm/submit.sh

The chains run about five times as many iterations within the same budget, so
the iteration caps are five times experiments_budget.py's (only upper limits),
and each chain takes about three to four times as long as the thesis HINTS
chain did: TIME is longer to match.
"""
import math

from experiments_budget import SS_PROPS, entry, iteration_cap

FUSED_CAP = 5


def experiments(datasets):
    out = []
    for dataset in datasets:
        for ss_prop in SS_PROPS[dataset]:
            e = entry(dataset, "HINTS", ss_prop, levels=int(math.log2(round(1 / ss_prop))),
                      branching=2, fused_buckets=True)
            e["iters"] = FUSED_CAP * iteration_cap(dataset, "HINTS", ss_prop)
            out.append(e)
    return out


EXPERIMENTS = experiments(("susy", "digits", "sensorless"))
