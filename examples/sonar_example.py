import copy
import json
import numpy as np
from discretesampling.base.random import RandomInt
import discretesampling.domain.spectrum as spec
from discretesampling.base.algorithms.rjmcmc import DiscreteVariableRJMCMC
from discretesampling.base.algorithms.continuous_proposals import sample_offsets, forward_grid_search, reverse_grid_search
from discretesampling.base.stan_model import stan_model

stan_model_path = "examples/stanmodels/linear_array.stan"
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
class continuous_proposal():
    def __init__(self):
        # grid search parameters
        self.grid_size = 1000
        self.min_vals = [0]
        self.max_vals = [np.pi]
        self.inds = [0]

        # store these values here so that they can be used when eval() is called (for efficiency, so we don't have to
        # re-compute many expensive logprob evaluations)
        self.current_discrete = None
        self.current_continuous = None
        self.proposed_discrete = None
        self.proposed_continuous = None
        self.forward_logprob = None
        self.to_remove = None  # don't think there's any way to work out the reverse proposal without this?

    def sample(self, x, params, y, rng):

        dim_x = x.value
        dim_y = y.value

        self.current_discrete = x
        self.proposed_discrete = y
        self.current_continuous = params

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
            offsets = sample_offsets(self.grid_size, self.min_vals, self.max_vals)
            [params_temp, forward_logprob] = forward_grid_search(data_function, model, self.grid_size, self.min_vals, self.max_vals, offsets, self.inds, params, y)

        elif (dim_x > dim_y):
            # randomly choose one to remove
            to_remove = RandomInt(0, dim_x-1).eval()
            if dim_y > 0:
                theta = np.delete(theta, to_remove)
            else:
                theta = np.array([])
            forward_logprob = 1 / dim_x
            params_temp = np.hstack((theta, delta2))
            self.to_remove = to_remove

        self.proposed_continuous = params_temp
        self.forward_logprob = forward_logprob

        return params_temp

    def eval(self, x, params, y, p_params):
        move_logprob = None
        dim_x = x.value
        dim_y = y.value
        # first check that we're doing an eval for the previous proposal
        if self.current_discrete == x:
            if self.proposed_discrete == y:
                if np.array_equal(self.current_continuous, params):
                    if np.array_equal(self.proposed_continuous, p_params):
                        if (dim_y > dim_x):
                            # Birth move
                            # Add new components
                            reverse_logprob = 1 / dim_y
                        elif (dim_x > dim_y):
                            reverse_logprob = reverse_grid_search(data_function, model, self.grid_size, self.min_vals,
                                                                  self.max_vals, self.inds, params, x, self.to_remove)
                        move_logprob = reverse_logprob - self.forward_logprob

        if move_logprob is None:
            raise Exception("Proposal probability undefined. This can only be called directly after sampling a from the \
                            proposal.")

        return move_logprob

# initialise stan model
model = stan_model(stan_model_path)

rjmcmc = DiscreteVariableRJMCMC(
    spec.SpectrumDimension,
    spec.SpectrumDimensionTarget(3, 2),
    model,
    data_function,
    continuous_proposal(),
    "NUTS",
    False,
    False,
    0.5,
    spec.SpectrumDimensionInitialProposal(1),
)

with open(data_path, 'r') as f:
    json_data = (json.load(f))
    for key, value in json_data.items():
        stationary_data.append([key, value])

n_chains = 1
samples = [rjmcmc.sample(200) for c in range(n_chains)]

dims = [[x[0].value for x in chain] for chain in samples]

print("Done")
