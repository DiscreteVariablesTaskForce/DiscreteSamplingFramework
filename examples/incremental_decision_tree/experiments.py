"""
The list of experiments run_experiments.py runs.
"""

import math

# Shared by every entry, so the samplers differ only in what is being compared.
COMMON = dict(dataset="susy", store_every=1, max_tree_size=100)
RUNS_LONG = {
    "smc": dict(chains=10, particles=10, steps=800),
    "mcmc": dict(chains=10, iters=800)
    }

RUNS = {
    "smc": dict(chains=10, particles=10, steps=200),
    "mcmc": dict(chains=10, iters=200)
    }
# MH Experiments
EXPERIMENTS = [dict(samplers=[sampler], proposals=["MH"],
                    **COMMON, **run)
               for sampler, run in RUNS_LONG.items()]

# EXPERIMENTS = []
SS_PROPS = (1/16, 1/4,)

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
