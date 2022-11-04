import copy
import json
import numpy as np
from discretesampling.base.random import RandomInt
import discretesampling.domain.spectrum as spec
from discretesampling.base.algorithms.rjmcmc import DiscreteVariableRJMCMC
from discretesampling.base.algorithms.continuous_proposals import forward_grid_search, reverse_grid_search
from discretesampling.base.stan_model import stan_model

stan_model_path = "StanForRJMCMCProblems/linear_array.stan"
bridgestan_path = "bridgestan"
cmdstan_path = "cmdstan"
data_path = "examples/5_targets_noisy.data.json"

stationary_data = []


# What data should be passed to the stan model given discrete variable x?
def data_function(x):
    dim = x.value
    data = [["M", dim]]
    for d in stationary_data:
        data.append(d)
    return data


# What params should be evaluated if we've made a discrete move from x to y
# where params is the params vector for the model defined by x
def continuous_proposal(x, params, y, rng):
    # grid search parameters
    grid_size = 1000
    min_vals = [0]
    max_vals = [np.pi]
    inds = [0]

    dim_x = x.value
    dim_y = y.value

    move_logprob = 0
    params_temp = copy.deepcopy(params)

    if dim_x > 0:
        theta = params_temp[0:dim_x]
        delta2 = params_temp[dim_x]
    else:
        theta = np.array([])
        # initialise delta2
        delta2 = rng.normal(0, 1)
        params = np.asarray([delta2])

    if (dim_y > dim_x):
        # Birth move
        # Add new components
        [params_temp, forward_logprob] = forward_grid_search(data_function, model, grid_size, min_vals, max_vals, inds, params, y)
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
        reverse_logprob = reverse_grid_search(data_function, model, grid_size, min_vals, max_vals, inds, params_temp, x, to_remove)
        move_logprob = reverse_logprob - forward_logprob

        params_temp = np.hstack((theta, delta2))

    return params_temp, move_logprob


# initialise stan model
model = stan_model(stan_model_path, bridgestan_path, cmdstan_path)

rjmcmc = DiscreteVariableRJMCMC(
    spec.SpectrumDimension,
    spec.SpectrumDimensionInitialProposal(1),
    spec.SpectrumDimensionTarget(3, 2),
    model,
    data_function,
    continuous_proposal,
    "NUTS",
    update_probability=0.5
)

with open(data_path, 'r') as f:
    json_data = (json.load(f))
    for key, value in json_data.items():
        stationary_data.append([key, value])

n_chains = 1
samples = [rjmcmc.sample(200) for c in range(n_chains)]

dims = [[x[0].value for x in chain] for chain in samples]

print("Done")
