"""
The Chapter 3 MCMC comparison at equal cost: every sampler is given the same
budget of row reads per chain, as every sampler of Chapter 2 was given the same
budget of full evaluations, and stops there.

    RUN_ID=ch3_budget EXPERIMENTS=examples/incremental_decision_tree/experiments_budget.py \\
        CORES=64 TIME=6:00:00 EVAL_CORES=64 slurm/submit.sh

The states are stored at RECORDS evenly spaced row-read marks, so the records
of every sampler sit at the same costs, and every measure taken over the second
half of a run is over the same window of cost. `iters` is only an upper limit,
set at three times the iterations the budget is expected to take, from the
row reads per iteration the first runs (experiments.py) measured.

The move log is off: at these lengths it would hold millions of moves per
chain, and the per-iteration accept flags it is read against are not kept by a
run with a budget. The move outcomes of the first runs describe the same
mechanisms. Chain i of every sampler starts from the same tree, and chains 0-15
are the chains of the first runs, continued.

The two-level HINTS hierarchy is in experiments_budget_shapes.py, a run of its
own, since the file names do not carry the shape of the hierarchy, and hepmass
is in experiments_budget_hepmass.py, also a run of its own.
"""
import math

MODEL = dict(lam=15.0, min_samples_leaf=3, max_tree_size=100)

# Row reads per chain: on susy about one and a half times what the longest
# chains of the first runs read, on digits ten times; on sensorless about
# 3.4 million iterations of MH; on hepmass the same as susy, since its trees
# reach the size cap within about 10^9 row reads.
BUDGET = {"susy": 1e11, "digits": 5e8, "sensorless": 1e10, "hepmass": 1e11}
SS_PROPS = {"susy": (1 / 4, 1 / 8, 1 / 16, 1 / 32), "digits": (1 / 4, 1 / 8),
            "sensorless": (1 / 4, 1 / 8, 1 / 16),
            "hepmass": (1 / 4, 1 / 8, 1 / 16, 1 / 32)}
CHAINS = 64
RECORDS = 5_000

# Row reads per iteration, by (dataset, proposal, ss_prop); MH reads the same
# whatever ss_prop says. susy and digits: the median of the first runs.
# sensorless: the second half of one 40,000-iteration chain (6,000 for the
# sweeps), at the size cap. hepmass: the second half of one 3,000-iteration
# chain (300 for the sweeps), with 1/32 extrapolated from 1/16 as on susy;
# these are early in a run, so its caps get a wider margin (CAP_FACTOR).
READS_PER_ITERATION = {
    ("susy", "MH", None): 3.0e5,
    ("susy", "DA", 1 / 4): 9.7e4, ("susy", "DA", 1 / 8): 8.2e4,
    ("susy", "DA", 1 / 16): 5.5e4, ("susy", "DA", 1 / 32): 3.9e4,
    ("susy", "FlatHINTS", 1 / 4): 9.9e5, ("susy", "FlatHINTS", 1 / 8): 1.7e6,
    ("susy", "FlatHINTS", 1 / 16): 2.8e6, ("susy", "FlatHINTS", 1 / 32): 4.7e6,
    ("susy", "HINTS", 1 / 4): 2.1e6, ("susy", "HINTS", 1 / 8): 2.9e6,
    ("susy", "HINTS", 1 / 16): 3.3e6, ("susy", "HINTS", 1 / 32): 4.8e6,
    ("digits", "MH", None): 100,
    ("digits", "DA", 1 / 4): 61, ("digits", "DA", 1 / 8): 53,
    ("digits", "FlatHINTS", 1 / 4): 498, ("digits", "FlatHINTS", 1 / 8): 767,
    ("digits", "HINTS", 1 / 4): 1_002, ("digits", "HINTS", 1 / 8): 1_131,
    ("sensorless", "MH", None): 2.9e3,
    ("sensorless", "DA", 1 / 4): 1.3e3, ("sensorless", "DA", 1 / 8): 620,
    ("sensorless", "DA", 1 / 16): 670,
    ("sensorless", "FlatHINTS", 1 / 4): 1.2e4, ("sensorless", "FlatHINTS", 1 / 8): 1.8e4,
    ("sensorless", "FlatHINTS", 1 / 16): 3.0e4,
    ("sensorless", "HINTS", 1 / 4): 1.9e4, ("sensorless", "HINTS", 1 / 8): 3.1e4,
    ("sensorless", "HINTS", 1 / 16): 3.3e4,
    ("hepmass", "MH", None): 2.3e5,
    ("hepmass", "DA", 1 / 4): 3.5e5, ("hepmass", "DA", 1 / 8): 1.25e5,
    ("hepmass", "DA", 1 / 16): 1.4e5, ("hepmass", "DA", 1 / 32): 1.0e5,
    ("hepmass", "FlatHINTS", 1 / 4): 3.0e6, ("hepmass", "FlatHINTS", 1 / 8): 4.2e6,
    ("hepmass", "FlatHINTS", 1 / 16): 3.5e6, ("hepmass", "FlatHINTS", 1 / 32): 6.0e6,
    ("hepmass", "HINTS", 1 / 4): 3.9e6, ("hepmass", "HINTS", 1 / 8): 4.3e6,
    ("hepmass", "HINTS", 1 / 16): 6.6e6, ("hepmass", "HINTS", 1 / 32): 9.6e6,
}
CAP_FACTOR = {"susy": 3, "digits": 3, "sensorless": 3, "hepmass": 5}
# The 2 x 4 hierarchy at 1/16 read about as much per iteration as the binary
# one at 1/16 did.
READS_PER_ITERATION[("susy", "HINTS2x4", 1 / 16)] = 3.2e6


def iteration_cap(dataset, key, ss_prop):
    """CAP_FACTOR times the iterations the budget is expected to take."""
    rate = READS_PER_ITERATION[(dataset, key, ss_prop)]
    return int(math.ceil(CAP_FACTOR[dataset] * BUDGET[dataset] / rate / 1_000) * 1_000)


def entry(dataset, proposal, ss_prop=None, key=None, **knobs):
    out = dict(dataset=dataset, samplers=["mcmc"], proposals=[proposal],
               chains=CHAINS, budget=BUDGET[dataset], records=RECORDS,
               iters=iteration_cap(dataset, key or proposal, ss_prop),
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


# hepmass is in experiments_budget_hepmass.py, a run of its own: splitting it
# needs more memory than the other datasets, and evaluating it takes longer.
EXPERIMENTS = experiments(("susy", "digits", "sensorless"))
