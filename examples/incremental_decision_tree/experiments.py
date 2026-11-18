"""
The list of experiments run_experiments.py runs: the sampler comparison of
Chapters 3 (MCMC) and 4 (SMC).

Two regimes:
- susy (4M training rows): the likelihood swamps the prior and trees grow to
  max_tree_size, so this is the cost and burn-in comparison.
- digits (1,437 training rows): the posterior sits well below the cap (about
  55-65 nodes), so this is the accuracy and mixing comparison.

Every chain and particle starts from the prior, the default start. The runs are
long enough for every sampler to pass a common budget of row reads, and the
analysis compares them at that budget: row_reads is recorded every iteration.
The long MH reference chains are in experiments_reference.py, a run of their
own, since they would write the same files as the MH runs here.
"""
import math

# The model and its prior: the same for every entry here and in the reference.
MODEL = dict(lam=15.0, min_samples_leaf=3, max_tree_size=100)

# Subsample fractions. The smaller ones only where there are rows enough to
# fill their blocks: digits' 1,437 rows give 180 per block at 1/8 already.
SS_PROPS = {"susy": (1 / 4, 1 / 8, 1 / 16, 1 / 32),
            "digits": (1 / 4, 1 / 8)}

# MCMC iterations, per dataset: (MH and DA, FlatHINTS and HINTS). A sweep costs
# several moves, so the sweeping proposals get fewer iterations for about the
# same row reads.
MCMC_ITERS = {"susy": (50_000, 10_000), "digits": (100_000, 25_000)}
# SMC steps, likewise: (MH and DA, FlatHINTS and HINTS).
SMC_STEPS = (2_000, 1_000)
# MCMC chains, or independent SMC runs: one core each. 16, as Chapter 2 ran
# each configuration, so run-to-run medians and R-hat read the same way.
CHAINS = 16
PARTICLES = 64

# Stored trees per chain, about: evaluation routes every stored tree through
# the test split, so storing every iteration of a long run costs more to
# evaluate than it did to sample.
MCMC_RECORDS = 5_000
SMC_RECORDS = 100     # steps, each holding all PARTICLES trees


def runs(dataset, proposal):
    """The MCMC and the SMC settings of one (dataset, proposal)."""
    sweeping = proposal in ("FlatHINTS", "HINTS")
    iters = MCMC_ITERS[dataset][sweeping]
    steps = SMC_STEPS[sweeping]
    return {
        "mcmc": dict(chains=CHAINS, iters=iters,
                     store_every=max(1, iters // MCMC_RECORDS)),
        "smc": dict(chains=CHAINS, particles=PARTICLES, steps=steps,
                    store_every=max(1, steps // SMC_RECORDS)),
    }


EXPERIMENTS = []
for dataset in ("susy", "digits"):
    for sampler, run in runs(dataset, "MH").items():
        EXPERIMENTS.append(dict(dataset=dataset, samplers=[sampler], proposals=["MH"],
                                **MODEL, **run))
    for proposal in ("DA", "FlatHINTS", "HINTS"):
        for ss_prop in SS_PROPS[dataset]:
            # levels and branching are HINTS-only knobs; validate_cfg refuses
            # them on an experiment that does not run HINTS. Binary levels down
            # to blocks of ss_prop: 2 levels at 1/4, ..., 5 at 1/32.
            knobs = (dict(levels=int(math.log2(round(1 / ss_prop))), branching=2)
                     if proposal == "HINTS" else {})
            for sampler, run in runs(dataset, proposal).items():
                EXPERIMENTS.append(dict(dataset=dataset, samplers=[sampler],
                                        proposals=[proposal], ss_prop=ss_prop,
                                        **knobs, **MODEL, **run))
