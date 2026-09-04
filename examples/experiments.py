"""
The list of experiments run_experiments.py runs.

Edit EXPERIMENTS below, then:

    python examples/run_experiments.py --list
    python examples/run_experiments.py

Each entry is a dict of overrides on top of sampler_diagnostics.DEFAULT_CFG --
see that dict for every field and its default (dataset, samplers, proposals,
chains, iters, steps, particles, metric_every, record_moves, ss_prop,
min_data, lam, min_samples_leaf, max_tree_size, jobs). Leave a field out to
keep its default.

An optional 'name' controls the .h5 filenames this experiment writes
(<name>_<sampler>_<proposal>.h5); it defaults to the experiment's dataset, so
two experiments on the same dataset need distinct names or run_experiments.py
refuses to run (one would silently overwrite the other's results).
"""

EXPERIMENTS = [
    # The defaults: wine, every sampler, every proposal, 4 chains.
    dict(dataset="wine"),

    # A coarser subsample on the same dataset -- needs its own name since
    # "wine" is already taken above.
    dict(dataset="wine", name="wine_ss25_md50",
         ss_prop=0.25, min_data=50),

    # A bigger, slower dataset with fewer chains to keep the sweep's total
    # runtime down.
    dict(dataset="digits", chains=2, iters=8_000),

    # Uncomment for a full-scale run -- covtype is slow, so this is left off
    # by default.
    # dict(dataset="covtype", name="covtype_hints", iters=20_000,
    #      metric_every=50, chains=4, samplers=["mcmc"], proposals=["HINTS"]),
]
