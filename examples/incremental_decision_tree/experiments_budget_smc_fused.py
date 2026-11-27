"""
The HINTS runs of experiments_budget_smc.py again, with fused_buckets (which
implies lazy_counts): every sweep is the one the thesis code makes, so every
SMC run moves its particles exactly as before, but each sweep's block
bookkeeping costs about a fifth of the row reads (see proposals/hints.py and
Results/overnight/REPORT.md). A run of its own because the file names do not
carry the option; MH, DA and FlatHINTS are not rerun.

    RUN_ID=ch4_smc_budget_fused EXPERIMENTS=examples/incremental_decision_tree/experiments_budget_smc_fused.py \\
        CORES=16 TIME=12:00:00 EVAL_CORES=32 slurm/submit.sh

About four to five times as many steps fit in the same budget, so the step caps
are five times experiments_budget_smc.py's (only upper limits).
"""
import math

from experiments_budget import SS_PROPS
from experiments_budget_smc import entry, step_cap

FUSED_CAP = 5


def experiments(datasets):
    out = []
    for dataset in datasets:
        for ss_prop in SS_PROPS[dataset]:
            e = entry(dataset, "HINTS", ss_prop, levels=int(math.log2(round(1 / ss_prop))),
                      branching=2, fused_buckets=True)
            e["steps"] = FUSED_CAP * step_cap(dataset, "HINTS", ss_prop)
            out.append(e)
    return out


EXPERIMENTS = experiments(("susy", "digits", "sensorless"))
