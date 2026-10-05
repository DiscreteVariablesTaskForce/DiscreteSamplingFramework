"""
The list of experiments run_experiments.py runs.
"""

import math

# Shared by every entry, so the samplers differ only in what is being compared.
COMMON = dict(dataset="covtype", store_every=1, max_tree_size=160)
RUNS = {"mcmc": dict(chains=10, iters=10_000),
        "smc": dict(chains=10, particles=20, steps=2_000)}

# MH Experiments
EXPERIMENTS = [dict(name=f"covtype_{sampler}", samplers=[sampler], proposals=["MH"],
                    **COMMON, **run)
               for sampler, run in RUNS.items()]

proposals = ["DA", "FlatHINTS", "HINTS"]

for prop in proposals:
    for ss_prop in (0.125, 0.25, 0.5):
        levels = int(math.log2(round(1 / ss_prop)))
        for sampler, run in RUNS.items():
            EXPERIMENTS.append(dict(
                name=f"covtype_ss{ss_prop * 1000:.0f}_{sampler}", samplers=[sampler],
                proposals=[prop], ss_prop=ss_prop,
                levels=levels, branching=2, **COMMON, **run))
