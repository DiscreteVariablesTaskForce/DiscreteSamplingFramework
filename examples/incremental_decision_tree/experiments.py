"""
The list of experiments run_experiments.py runs.
"""

import math

# Shared by every entry, so the samplers differ only in what is being compared.
COMMON = dict(dataset="covtype", store_every=1, max_tree_size=80)
RUNS_LONG = {"mcmc": dict(chains=8, iters=16_000),
             "smc": dict(chains=8, particles=8, steps=8_000)}

RUNS = {"mcmc": dict(chains=8, iters=2_000),
        "smc": dict(chains=8, particles=8, steps=1_000)}
# MH Experiments
EXPERIMENTS = [dict(samplers=[sampler], proposals=["MH"],
                    **COMMON, **run)
               for sampler, run in RUNS_LONG.items()]

EXPERIMENTS = []
proposals = ["DA", "FlatHINTS", "HINTS"]

for prop in proposals:
    for ss_prop in (0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5):
        # levels and branching are HINTS-only knobs; validate_cfg refuses them
        # on an experiment that does not run HINTS.
        knobs = (dict(levels=int(math.log2(round(1 / ss_prop))), branching=2)
                 if prop == "HINTS" else {})
        if prop == "DA":
            runs_list = RUNS_LONG
        else:
            runs_list = RUNS
        for sampler, run in runs_list.items():
            EXPERIMENTS.append(dict(
                samplers=[sampler],
                proposals=[prop], ss_prop=ss_prop,
                **knobs, **COMMON, **run))
