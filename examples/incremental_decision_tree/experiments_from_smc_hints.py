"""
The experiments.py sweep on susy, started from where SMC-HINTS got to in one
step instead of from a stump.

    python examples/incremental_decision_tree/run_experiments.py \\
        --file examples/incremental_decision_tree/experiments_from_smc_hints.py --list
    python examples/incremental_decision_tree/run_experiments.py \\
        --file examples/incremental_decision_tree/experiments_from_smc_hints.py \\
        --run-id susy_from_hints

Every chain and every particle of every experiment here starts from the one
tree INIT_TREE. The first step of Results/susy_results/susy_smc_HINTS_1p5625.h5
is 20 particles per run, each a different tree, but one of them carries all of
the weight (1.000, tens of thousands of log units clear of the next), so that
particle is what the step put out. Which one it is depends on the run:

    runs 0-5    a 12-node tree, log-likelihood -2,352,081, 0.673 train accuracy
    runs 6-19   a 12-node tree, log-likelihood -2,301,893, 0.707 train accuracy

SOURCE_RUN picks the second: it is the output of 14 of the 20 runs and the
better tree. Set it to 0 for the first.
"""
import math
import os

from results_io import stored_tree

SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
                      "Results", "susy_results", "susy_smc_HINTS_1p5625.h5")
SOURCE_RUN = 6
# record=0 is the first SMC step; no particle named, so the heaviest one.
INIT_TREE = stored_tree(SOURCE, run=SOURCE_RUN, record=0)

# Shared by every entry, so the samplers differ only in what is being compared.
# init_tree_name tags the files: susy_mcmc_DA_1p5625_init-hints-step0.h5.
COMMON = dict(dataset="susy", store_every=1, max_tree_size=100,
              init_tree=INIT_TREE, init_tree_name="hints-step0")

# The run lengths susy_results was sampled with, so the two are comparable.
RUNS_LONG = {"mcmc": dict(chains=20, iters=32_000),
             "smc": dict(chains=20, particles=20, steps=8_000)}
RUNS = {"mcmc": dict(chains=20, iters=2_000),
        "smc": dict(chains=20, particles=20, steps=500)}

# susy_results also has 0.0625 and 0.25.
SS_PROPS = (0.0625, 0.25)

# MH Experiments
EXPERIMENTS = [dict(samplers=[sampler], proposals=["MH"],
                    **COMMON, **run)
               for sampler, run in RUNS_LONG.items()]

for prop in ("DA", "FlatHINTS", "HINTS"):
    for ss_prop in SS_PROPS:
        # levels and branching are HINTS-only knobs; validate_cfg refuses them
        # on an experiment that does not run HINTS.
        knobs = (dict(levels=int(math.log2(round(1 / ss_prop))), branching=2)
                 if prop == "HINTS" else {})
        runs_list = RUNS_LONG if prop == "DA" else RUNS
        for sampler, run in runs_list.items():
            EXPERIMENTS.append(dict(
                samplers=[sampler],
                proposals=[prop], ss_prop=ss_prop,
                **knobs, **COMMON, **run))
