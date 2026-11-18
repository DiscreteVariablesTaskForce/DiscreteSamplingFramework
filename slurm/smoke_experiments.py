"""
A few seconds of every sampler and two proposals on wine: submit this first on
a new cluster set-up, to find a missing module, a broken virtualenv or a wrong
path before the long runs do.

    RUN_ID=smoke EXPERIMENTS=slurm/smoke_experiments.py CORES=2 TIME=0:20:00 \
    EVAL_CORES=2 EVAL_TIME=0:20:00 slurm/submit.sh
"""

COMMON = dict(dataset="wine", store_every=1, max_tree_size=20, chains=2,
              iters=50, steps=5, particles=4, ss_prop=0.25)

EXPERIMENTS = [dict(samplers=[sampler], proposals=[proposal], **COMMON)
               for sampler in ("mcmc", "smc") for proposal in ("MH", "HINTS")]
