import copy
import numpy as np
from discretesampling.base.random import RandomInt

import discretesampling.domain.spectrum as spec

from discretesampling.base.algorithms.rjmcmc import DiscreteVariableRJMCMC

from scipy.stats import multivariate_normal
from discretesampling.base.stan_model import stan_model

stan_model_path = "examples/stanmodels/mixturemodel.stan"


# What data should be passed to the stan model given discrete variable x?
def data_function(x):
    dim = x.value
    data = [["K", dim], ["N", 20], ["y", [3, 4, 5, 4, 5, 3, 4, 5, 6, 3, 4, 5, 15, 16, 15, 17, 16, 18, 17, 14]]]
    return data


# What params should be evaluated if we've made a discrete move from x to y
# where params is the params vector for the model defined by x
def continuous_proposal(x, params, y, rng):
    dim_x = x.value
    dim_y = y.value

    move_logprob = 0

    params_temp = copy.deepcopy(params)
    if dim_x > 1:
        theta = params_temp[0:(dim_x-1)]  # k-simplex parameterised as K-1 unconstrained
        mu = params_temp[(dim_x-1):(2*dim_x-1)]
        sigma = params_temp[(2*dim_x-1):len(params_temp)]
    else:
        theta = []  # One component, mixing component is undefined
        mu = params_temp[0:dim_x]
        sigma = params_temp[dim_x:len(params_temp)]

    if (dim_y > dim_x):
        # Birth move
        # Add new components
        n_new_components = dim_y - dim_x

        if dim_x > 0:
            theta_new = rng.multivariate_normal(np.zeros(n_new_components), np.identity(n_new_components))
            mu_new = rng.multivariate_normal(np.zeros(n_new_components), np.identity(n_new_components))
            sigma_new = rng.multivariate_normal(np.zeros(n_new_components), np.identity(n_new_components))

            mvn = multivariate_normal(np.zeros(n_new_components), np.identity(n_new_components))
            forward_logprob = mvn.logpdf(theta_new) + mvn.logpdf(mu_new) + mvn.logpdf(sigma_new)
        else:
            # initial proposal
            theta_new = rng.multivariate_normal(np.zeros(n_new_components-1), np.identity(n_new_components-1))
            mu_new = rng.multivariate_normal(np.zeros(n_new_components), np.identity(n_new_components))
            sigma_new = rng.multivariate_normal(np.zeros(n_new_components), np.identity(n_new_components))

            mvn = multivariate_normal(np.zeros(n_new_components), np.identity(n_new_components))
            mvn_theta = multivariate_normal(np.zeros(n_new_components-1), np.identity(n_new_components-1))
            forward_logprob = mvn_theta.logpdf(theta_new) + mvn.logpdf(mu_new) + mvn.logpdf(sigma_new)
        reverse_logprob = np.log(1/dim_y)
        move_logprob = forward_logprob - reverse_logprob

        params_temp = np.concatenate((np.array(theta), theta_new, np.array(mu), mu_new, np.array(sigma), sigma_new))

        return params_temp, move_logprob
    elif (dim_x > dim_y):
        # randomly choose one to remove
        to_remove = RandomInt(0, dim_x-1).eval()
        # birth of component could happen multiple ways (i.e. different n_new_components)
        # so I think the reverse_logprob will only be approximate - seems like there might
        # be a proof for the summed pdf values of the series of Gaussians that we can use?
        mvn = multivariate_normal(0, 1)
        reverse_logprob = mvn.logpdf(theta[to_remove]) + mvn.logpdf(mu[to_remove]) + mvn.logpdf(sigma[to_remove])
        if dim_y > 1:
            theta = np.delete(theta, to_remove)
        else:
            theta = np.array([])

        mu = np.delete(mu, to_remove)
        sigma = np.delete(sigma, to_remove)

        forward_logprob = np.log(1/dim_x)
        move_logprob = forward_logprob - reverse_logprob

        params_temp = np.concatenate((theta, mu, sigma))

    return list(params_temp), move_logprob


# initialise stan model
model = stan_model(stan_model_path)

rjmcmc = DiscreteVariableRJMCMC(
    spec.SpectrumDimension,
    spec.SpectrumDimensionInitialProposal(10),  # Initial proposal uniform(0,20)
    spec.SpectrumDimensionTarget(3, 2),  # NB prior on number of mixture components
    model,
    data_function,
    continuous_proposal,
    "NUTS",
    update_probability=0.5
)

n_chains = 4
samples = [rjmcmc.sample(1000) for c in range(n_chains)]

dims = [[x[0].value for x in chain] for chain in samples]

print("Done")
