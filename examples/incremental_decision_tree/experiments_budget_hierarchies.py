"""
The shape of the HINTS hierarchy on the two large datasets, SUSY and HEPMASS,
at the budget of experiments_budget.py, with DA and Flat HINTS at the 64-block
split beside it.

Hierarchies are written as in Chapter 2, [m_0, m_1, ...]: m_0 blocks at each
unit vertex (always 1 here) and the branch factor of each level above it, so
[1,8] is eight unit vertices under the global vertex with no intermediate
levels, and [1,2,2,2] is the binary hierarchy over the same eight. HINTS has one
branch factor m for every level (levels=H, branching=m below), so [1,m,...,m]
has m**H unit vertices, each holding a fraction 1/m**H of the rows below the
subtree root. These are every such hierarchy with up to 64 unit vertices that
the budget runs have not already covered (the binary [1,2,2] to [1,2,2,2,2,2]
are in experiments_budget.py and experiments_budget_hepmass.py):

    unit vertices   hierarchies
    4               [1,4]
    8               [1,8]
    16              [1,16], [1,4,4]
    32              [1,32]
    64              [1,2,2,2,2,2,2], [1,4,4,4], [1,8,8], [1,64],
                    with DA and Flat HINTS at 1/64

The file names carry ss_prop but not the shape, so no two experiments of one run
may share a dataset and ss_prop. The shapes are therefore split over four runs,
one per GROUPS entry. This file is the first; the other three are the thin
files beside it:

    RUN_ID=ch3_budget_split64 \\
        EXPERIMENTS=examples/incremental_decision_tree/experiments_budget_hierarchies.py ...
    RUN_ID=ch3_budget_one_level \\
        EXPERIMENTS=examples/incremental_decision_tree/experiments_budget_hierarchies_one_level.py ...
    RUN_ID=ch3_budget_branch4 \\
        EXPERIMENTS=examples/incremental_decision_tree/experiments_budget_hierarchies_branch4.py ...
    RUN_ID=ch3_budget_branch8 \\
        EXPERIMENTS=examples/incremental_decision_tree/experiments_budget_hierarchies_branch8.py ...

each with the HEPMASS settings, since every run holds HEPMASS:
    PREP_CORES=4 CORES=64 TIME=8:00:00 EVAL_CORES=64 EVAL_TIME=24:00:00 slurm/submit.sh

[1,4,4] on SUSY was run before (experiments_budget_shapes.py, ch3_budget_2x4), but
before HINTS dropped levels by the branching factor it was given rather than by
2, so on a subtree too small for 16 blocks of min_data rows it kept 16 smaller
ones. It is run again here so that every hierarchy of a fixed branching factor
uses the same rule; the binary shapes, DA and Flat HINTS are unchanged by it.

MH is not rerun: chain i of every sampler starts from the same tree, so the MH
chains of ch3_budget (SUSY) and ch3_budget_hepmass are the reference, and the
analysis takes every run id at once:
    python examples/incremental_decision_tree/analyse_results.py \\
        --run-ids ch3_budget ch3_budget_hepmass ch3_budget_split64 \\
                  ch3_budget_one_level ch3_budget_branch4 ch3_budget_branch8
"""
import math

from experiments_budget import BUDGET, CAP_FACTOR, CHAINS, MODEL, RECORDS

DATASETS = ("susy", "hepmass")

# Lower bounds on the row reads per iteration, so that the iteration cap
# (CAP_FACTOR times the iterations the budget would take at this rate) can only
# be reached after the budget. The budget runs read at least 1.8e4 (DA, SUSY,
# 1/32) and 1.3e6 (Flat HINTS and HINTS, SUSY, 1/4) per iteration, and a smaller
# split reads less for DA and more for the sweeps. The cap only bounds the loop;
# nothing is allocated per iteration.
LOW_READS_PER_ITERATION = {"DA": 5e3, "FlatHINTS": 5e5, "HINTS": 5e5}


def iteration_cap(dataset, proposal):
    rate = LOW_READS_PER_ITERATION[proposal]
    return int(math.ceil(CAP_FACTOR[dataset] * BUDGET[dataset] / rate / 1_000) * 1_000)


def entry(dataset, proposal, ss_prop, **knobs):
    return dict(dataset=dataset, samplers=["mcmc"], proposals=[proposal],
                ss_prop=ss_prop, chains=CHAINS, budget=BUDGET[dataset],
                records=RECORDS, iters=iteration_cap(dataset, proposal),
                record_moves=False, init_tree_size=None, **MODEL, **knobs)


def hints(dataset, levels, branching):
    """HINTS with `levels` levels of `branching` children each, so its unit
    blocks are 1/branching**levels of the rows below the subtree root."""
    return entry(dataset, "HINTS", 1 / branching ** levels,
                 levels=levels, branching=branching)


GROUPS = {
    # The 64-block split: DA, Flat HINTS and the binary hierarchy [1,2,2,2,2,2,2].
    "split64": [e for d in DATASETS for e in (entry(d, "DA", 1 / 64),
                                              entry(d, "FlatHINTS", 1 / 64),
                                              hints(d, 6, 2))],
    # [1,m]: m unit vertices under the global vertex, with no intermediate
    # levels. This differs from Flat HINTS only in scaling the block counts and
    # tempering the screens.
    "one_level": [hints(d, 1, m) for d in DATASETS for m in (4, 8, 16, 32, 64)],
    # Branch factor 4: [1,4,4] (16 unit vertices) and [1,4,4,4] (64).
    "branch4": [hints(d, levels, 4) for d in DATASETS for levels in (2, 3)],
    # Branch factor 8: [1,8,8] (64 unit vertices).
    "branch8": [hints(d, 2, 8) for d in DATASETS],
}

EXPERIMENTS = GROUPS["split64"]
