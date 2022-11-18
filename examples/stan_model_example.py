from discretesampling.base.stan_model import stan_model
from numpy.random import default_rng
import numpy as np
from scipy.stats import multivariate_normal
from discretesampling.base.random import Random
import math

import matplotlib.pyplot as plt

mixturemodel = stan_model(
    "examples/stanmodels/mixturemodel.stan"
)

niters = 500
sigma = 1

data = [["K", 5], ["N", 20], ["y", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]]

# 5 mixture component locations and sigmas, 5-simplex (described by four-element unconstrained vector)
param_length = mixturemodel.num_unconstrained_parameters(data)
print("Param_length:", param_length)

mu = [0 for i in range(param_length)]
sigma = np.identity(param_length)

rng = default_rng()
init = rng.multivariate_normal(mu, sigma*10)
samples = [init]
current = init

# MCMC
for i in range(niters):
    proposed = current + rng.multivariate_normal(mu, sigma)
    current_target = mixturemodel.eval(data, current)[0]
    proposed_target = mixturemodel.eval(data, proposed)[0]
    forward_logprob = multivariate_normal.logpdf(proposed, mean=current, cov=sigma)
    reverse_logprob = multivariate_normal.logpdf(current, mean=proposed, cov=sigma)
    log_acceptance_ratio = proposed_target - current_target + reverse_logprob - forward_logprob
    if log_acceptance_ratio > 0:
        log_acceptance_ratio = 0
    acceptance_probability = min(1, math.exp(log_acceptance_ratio))
    q = Random().eval()
    if (q < acceptance_probability):
        current = proposed
    samples.append(current)


fig, axes = plt.subplots(4, 4)
for j in range(param_length):
    ax = axes.flat[j]
    ax.plot([i for i in range(len(samples))], [x[j] for x in samples])

plt.savefig("mixturemodel-trace.png")
print("Done")
