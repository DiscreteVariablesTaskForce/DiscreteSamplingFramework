"""
Long MH chains on each dataset of experiments.py, as the reference the
sampler comparison is measured against: posterior tree sizes, log-posterior
range (for burn-in), predictive metrics.

They write the same file names as experiments.py's MH runs, so they go in a
run of their own:

    RUN_ID=reference EXPERIMENTS=examples/incremental_decision_tree/experiments_reference.py \
        slurm/submit.sh
"""
from experiments import CHAINS, MCMC_RECORDS, MODEL   # same model, same prior

REFERENCE_ITERS = {"susy": 200_000, "digits": 500_000}

EXPERIMENTS = [dict(dataset=dataset, samplers=["mcmc"], proposals=["MH"],
                    chains=CHAINS, iters=iters,
                    store_every=max(1, iters // MCMC_RECORDS), **MODEL)
               for dataset, iters in REFERENCE_ITERS.items()]
