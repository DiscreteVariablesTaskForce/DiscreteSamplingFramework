import copy
import json
import numpy as np
from discretesampling.base.random import RandomInt

import discretesampling.domain.spectrum as spec

from discretesampling.base.algorithms.rjmcmc import DiscreteVariableRJMCMC

stan_model_path = "StanForRJMCMCProblems/linear_array.stan"
redding_stan_path = "redding-stan/redding-stan"
data_path = "examples/5_targets_noisy.data.json"

stationary_data = []

# What's the dimensionality of the model represented by discrete variable x?
def continuous_dim_function(x):
    dim = x.value
    return dim + 1


# What data should be passed to the stan model given discrete variable x?
def data_function(x):
    dim = x.value
    data = [["M", dim]]
    for d in stationary_data:
        data.append(d)
    return data


# What params should be evaluated if we've made a discrete move from x to y
# where params is the params vector for the model defined by x
def continuous_proposal(x, params, y):
    # grid search parameters
    grid_size = 200
    min_vals = [0]
    max_vals = [np.pi]
    inds = [0]

    dim_x = x.value
    dim_y = y.value

    move_logprob = 0
    params_temp = copy.deepcopy(params)
    theta = params_temp[0:dim_x]
    delta2 = params_temp[dim_x]

    if (dim_y > dim_x):
        # Birth move
        # Add new components
        [params_temp, forward_logprob] = rjmcmc.forward_grid_search(grid_size, min_vals, max_vals, inds, params, y)
        reverse_logprob = 1 / dim_y
        move_logprob = reverse_logprob - forward_logprob

    elif (dim_x > dim_y):
        # randomly choose one to remove
        to_remove = RandomInt(0, dim_x-1).eval()
        if dim_y > 0:
            theta = np.delete(theta, to_remove)
        else:
            theta = np.array([])

        forward_logprob = 1 / dim_x
        reverse_logprob = rjmcmc.reverse_grid_search(grid_size, min_vals, max_vals, inds, params_temp, x, to_remove)
        move_logprob = reverse_logprob - forward_logprob

        params_temp = np.hstack((theta, delta2))

    return params_temp, move_logprob


rjmcmc = DiscreteVariableRJMCMC(
    spec.SpectrumDimension,
    spec.SpectrumDimensionInitialProposal(1),
    spec.SpectrumDimensionTarget(3, 2),
    "linear_array",
    "redding-stan",
    data_function,
    continuous_dim_function,
    continuous_proposal,
    "NUTS",
    update_probability=0.5
)

with open(data_path, 'r') as f:
    json_data = (json.load(f))
    for key, value in json_data.items():
        stationary_data.append([key, value])

n_chains = 4
samples = [rjmcmc.sample(1000) for c in range(n_chains)]

dims = [[x[0].value for x in chain] for chain in samples]

print("Done")
