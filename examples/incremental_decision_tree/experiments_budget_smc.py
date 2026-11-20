"""
The Chapter 4 SMC comparison at equal cost: 16 independent SMC runs of 64
particles per configuration, each run given the same budget of row reads as
one MCMC chain of experiments_budget.py, so that an SMC run and an MCMC chain
can be compared at equal cost.

    RUN_ID=ch4_smc_budget EXPERIMENTS=examples/incremental_decision_tree/experiments_budget_smc.py \\
        CORES=16 TIME=6:00:00 EVAL_CORES=32 slurm/submit.sh

Each run stores its population, with the weights, at RECORDS evenly spaced
row-read marks, and keeps its per-step columns (the ESS above all) at every
step. `steps` is only an upper limit, set from the row reads per step of the
first SMC runs (experiments.py) on susy and digits, and from 64 times the row
reads per MCMC iteration of experiments_budget.py on sensorless and hepmass,
with a wider margin for those. The move log is off, as for the MCMC runs.
HEPMASS is in experiments_budget_smc_hepmass.py, a run of its own.
"""
import math

from experiments_budget import BUDGET, MODEL, READS_PER_ITERATION, SS_PROPS

RUNS = 16
PARTICLES = 64
RECORDS = 100

# Row reads per SMC step, by (dataset, proposal, ss_prop): the median of the
# first SMC runs on susy and digits.
READS_PER_STEP = {
    ("susy", "MH", None): 2.88e7,
    ("susy", "DA", 1 / 4): 8.43e6, ("susy", "DA", 1 / 8): 5.23e6,
    ("susy", "DA", 1 / 16): 3.52e6, ("susy", "DA", 1 / 32): 2.97e6,
    ("susy", "FlatHINTS", 1 / 4): 8.51e7, ("susy", "FlatHINTS", 1 / 8): 1.38e8,
    ("susy", "FlatHINTS", 1 / 16): 2.17e8, ("susy", "FlatHINTS", 1 / 32): 3.43e8,
    ("susy", "HINTS", 1 / 4): 1.78e8, ("susy", "HINTS", 1 / 8): 2.11e8,
    ("susy", "HINTS", 1 / 16): 2.50e8, ("susy", "HINTS", 1 / 32): 3.26e8,
    ("digits", "MH", None): 6.69e3,
    ("digits", "DA", 1 / 4): 4.57e3, ("digits", "DA", 1 / 8): 3.73e3,
    ("digits", "FlatHINTS", 1 / 4): 4.14e4, ("digits", "FlatHINTS", 1 / 8): 5.38e4,
    ("digits", "HINTS", 1 / 4): 1.04e5, ("digits", "HINTS", 1 / 8): 1.16e5,
}
# Elsewhere, one step moves every particle once, so about PARTICLES MCMC
# iterations' worth of reads.
for (dataset, proposal, ss_prop), rate in READS_PER_ITERATION.items():
    if dataset in ("sensorless", "hepmass"):
        READS_PER_STEP[(dataset, proposal, ss_prop)] = PARTICLES * rate
CAP_FACTOR = {"susy": 3, "digits": 3, "sensorless": 5, "hepmass": 5}


def step_cap(dataset, proposal, ss_prop):
    """CAP_FACTOR times the steps the budget is expected to take."""
    rate = READS_PER_STEP[(dataset, proposal, ss_prop)]
    return int(math.ceil(CAP_FACTOR[dataset] * BUDGET[dataset] / rate / 100) * 100)


def entry(dataset, proposal, ss_prop=None, **knobs):
    out = dict(dataset=dataset, samplers=["smc"], proposals=[proposal],
               chains=RUNS, particles=PARTICLES, budget=BUDGET[dataset],
               records=RECORDS, steps=step_cap(dataset, proposal, ss_prop),
               record_moves=False, init_tree_size=None, **MODEL, **knobs)
    if ss_prop is not None:
        out["ss_prop"] = ss_prop
    return out


def experiments(datasets):
    """MH, and DA, FlatHINTS and HINTS at each ss_prop, on each dataset."""
    out = []
    for dataset in datasets:
        out.append(entry(dataset, "MH"))
        for proposal in ("DA", "FlatHINTS", "HINTS"):
            for ss_prop in SS_PROPS[dataset]:
                knobs = (dict(levels=int(math.log2(round(1 / ss_prop))), branching=2)
                         if proposal == "HINTS" else {})
                out.append(entry(dataset, proposal, ss_prop, **knobs))
    return out


EXPERIMENTS = experiments(("susy", "digits", "sensorless"))
