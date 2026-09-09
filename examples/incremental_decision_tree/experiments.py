"""
The list of experiments run_experiments.py runs.

Edit EXPERIMENTS below, then:

    python examples/incremental_decision_tree/run_experiments.py --list
    python examples/incremental_decision_tree/run_experiments.py
    python examples/incremental_decision_tree/run_experiments.py --index 0     # just the first one

and evaluate what came out afterwards -- sampling stores trees and computes no
metrics; evaluate_results.py turns the trees into metrics:

    python examples/incremental_decision_tree/evaluate_results.py --run-id <the run this printed>

Each entry is a dict of overrides on top of sampler_diagnostics.DEFAULT_CFG --
see that dict for every field and its default (dataset, samplers, proposals,
chains, iters, steps, particles, store_every, record_states, record_moves,
ss_prop, min_data, lam, min_samples_leaf, max_tree_size, jobs). Leave a field
out to keep its default.

An optional 'name' controls the .h5 filenames this experiment writes
(<name>_<sampler>_<proposal>.h5) and is what you pass to
`plot_diagnostics.py --name`; it defaults to the experiment's dataset, so two
experiments on the same dataset need distinct names or run_experiments.py
refuses to run (one would silently overwrite the other's results).


BUDGET-MATCHED MCMC vs SMC
--------------------------
The three covtype entries below all spend the same 500,000 proposal draws and
target evaluations per experiment unit, so a metric plotted against wallclock
compares like with like:

    mcmc      50 independent chains x 10,000 iterations   = 500,000
    smc       500 particles x 1,000 steps                 = 500,000 per run
    smc_50p   50 particles x 10,000 steps                 = 500,000 per run

The two SMC entries spend that budget differently and answer different
questions. `smc` is how SMC is normally configured -- a large population over
a short sequence. `smc_50p` matches the MCMC side element for element (same
population size, same number of steps), which leaves resampling and weighting
as the only difference between them; that is the controlled comparison, and
the one that isolates what resampling actually buys here.

Two caveats worth knowing before reading the results:

  * The target in this code is fixed across SMC steps -- there is no annealing
    or tempering sequence (see smc.py). Resampling earns its keep when a
    sampler is traversing a sequence of distributions; against a static target
    it mostly costs particle diversity, while concentrating effort on
    high-weight states. Expect SMC to look better on accuracy-per-second early
    and worse on anything measuring posterior coverage.
  * MCMC's 50 chains are independent, so the spread across them is a real
    Monte Carlo error. SMC's particles are coupled by resampling and its ESS
    collapses to ~1 in these runs, so its 500 particles are nowhere near 500
    independent samples. `chains` on the SMC entries is the number of repeats
    of the whole run, which is where its error bars come from -- at 4 repeats
    they cost 4x the matched budget. Drop to chains=1 for the strict budget
    match, at the price of having no band to plot.


RUNTIME
-------
Measured on covtype (464k train / 116k test rows), sampling cost per
iteration or step:

    MCMC   MH 2.0ms   DA 3.4ms   HINTS 17.0ms
    SMC    MH 108ms   DA 148ms   HINTS 1010ms      (at 30 particles; SMC cost
                                                    scales with `particles`)

That is now what a run costs. Storing a state is a few array copies over a
tree of at most `max_tree_size` nodes, and only *distinct* states are stored,
so `store_every=1` -- every iteration, which is what makes the per-iteration
curves dense -- adds a percent or two rather than dominating.

The evaluation is what used to be charged to the run, and it is now paid
separately, once, by evaluate_results.py:

    ~52.6ms per distinct ensemble, over both splits

For MCMC an ensemble is one tree, and a chain visits far fewer distinct trees
than it runs iterations (a rejection stays put), so a 10,000-iteration chain
costs a few seconds rather than ~9 minutes. For SMC an ensemble is the
particles at one step, but resampling collapses them to a handful of distinct
trees, so a step costs a small multiple of one tree rather than `particles`
times it. `--stride`, `--splits test` and `--jobs` cut it further.

So the thing to plan a walltime against is the sampling cost above, and the
knobs that matter are `iters`/`steps`, `particles`, and whether HINTS is in
`proposals` (7x the sampling cost of the other two).
"""

EXPERIMENTS = [
    dict(name="covtype_mcmc", dataset="covtype", samplers=["mcmc"],
         chains=5, iters=500, store_every=1),

    dict(name="covtype_smc", dataset="covtype", samplers=["smc"],
         chains=1, particles=10, steps=250, store_every=1),

    dict(name="covtype_smc_50p", dataset="covtype", samplers=["smc"],
         chains=1, particles=5, steps=500, store_every=1),

    # Quick smoke tests -- wine runs in seconds, for checking the pipeline
    # end to end before committing to a covtype run.
    # dict(dataset="wine", chains=4, iters=2_000, steps=20, particles=200),
    # dict(dataset="digits", name="digits_quick", chains=2, iters=5_000,
    #      steps=20, particles=200),
]
