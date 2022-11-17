from discretesampling.domain import spectrum
from discretesampling.base.algorithms import DiscreteVariableMCMC, DiscreteVariableSMC

import copy
import json
import itertools
import numpy as np
from discretesampling.base.random import RandomInt
import discretesampling.domain.spectrum as spec
import discretesampling.domain.reversible_jump as rj
from discretesampling.base.algorithms.continuous_proposals import sample_offsets, grid_search_logprobs, forward_grid_search, reverse_grid_search
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
class continuous_proposal():
    def __init__(self):
        # grid search parameters
        self.grid_size = 10
        self.min_vals = [0]
        self.max_vals = [np.pi]
        self.inds = [0]

    def sample(self, x, params, y, rng):

        dim_x = x.value
        dim_y = y.value

        params_temp = copy.deepcopy(params)

        logprob_vals = np.asarray([])
        offset_vals = np.asarray([])

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
            [params_temp, forward_logprob] = forward_grid_search(data_function, model, self.grid_size, self.min_vals, self.max_vals, offsets, self.inds, params[0:(dim_x + 1)], y)
            if len(params) > dim_x+1:
                if params[dim_x + 1] == 1:
                    logprob_vals = params_temp[-(dim_x + 1):-1]
                    offset_vals = params_temp[(dim_x + 2):-(dim_x + 1)]
                elif params[dim_x + 1] == -1:
                    logprob_vals = params_temp[-(dim_x + 2):-2]  # remove one logprob value as it's death -> birth
                    offset_vals = params_temp[(dim_x + 2):-(dim_x + 3)]
                else:
                    raise Exception("Something weird happened with the discrete moves.")
            last_move = 1  # flag birth
            params_temp = np.hstack((params_temp, last_move, offset_vals, offsets, logprob_vals, forward_logprob))  # store previous parameters for eval check in case we go through a series of discrete moves that lead back to the same point (expensive to work out forward_logprob again)

        elif (dim_x > dim_y):
            # randomly choose one to remove
            to_remove = RandomInt(0, dim_x-1).eval()
            if dim_y > 0:
                theta = np.delete(theta, to_remove)
            else:
                theta = np.array([])
            forward_logprob = 1 / dim_x
            if len(params) > dim_x+1:
                if params[dim_x + 1] == 1:
                    logprob_vals = params_temp[-(dim_x + 2):-2]  # remove one logprob value as it's birth -> death
                    offset_vals = params_temp[(dim_x + 2):-(dim_x + 3)]
                elif params[dim_x + 1] == -1:
                    logprob_vals = params_temp[-(dim_x + 2):-3]  # remove two logprob value as it's death -> death
                    offset_vals = params_temp[(dim_x + 2):-(dim_x + 3)]
                else:
                    raise Exception("Something weird happened with the discrete moves.")
            last_move = -1  # flag death
            params_temp = np.hstack((theta, delta2, last_move, offset_vals, logprob_vals, forward_logprob))

        return params_temp

    def eval(self, x, params, y, p_params):
        dim_x = x.value
        dim_y = y.value
        current_data = data_function(x)
        param_length = model.num_unconstrained_parameters(current_data)
        p_data = data_function(y)
        p_param_length = model.num_unconstrained_parameters(p_data)
        if dim_x == 0 and dim_y == 1:
            # initialisation (birth move)
            reverse_logprob = 1 / dim_y
            move_logprob = reverse_logprob - p_params[-1]
        elif dim_x == 0:
            # undefined move
            move_logprob = np.NINF
        elif np.array_equal(p_params[p_param_length:(p_param_length + param_length)], params[0:param_length]):
            if (dim_y == dim_x + 1):
                # Birth move
                num_matching = 0
                for n in range(0, param_length):
                    num_matching += len(np.where(p_params == params[n]))
                if num_matching == param_length:
                    # can reuse forward logprob
                    reverse_logprob = 1 / dim_y
                    move_logprob = reverse_logprob - p_params[-1]
                else:
                    # requires multiple moves
                    move_logprob = np.NINF
            elif (dim_y == dim_x + 1):
                # Death move
                num_matching = 0
                for n in range(0, p_param_length):
                    num_matching += len(np.where(params == p_params[n]))
                if num_matching == p_param_length:
                    # can reuse forward logprob
                    living = np.asarray([])
                    # work out index of component that was removed
                    for n in range(0, p_param_length):
                        ind = np.where(params == p_params[n])
                        living = np.hstack(living, ind)
                    all_inds = range(0, p_param_length)
                    for n in range(0, p_param_length):
                        if len(np.where(params == all_inds[n])) == 0:
                            death_ind = all_inds[n]
                    reverse_logprob = reverse_grid_search(data_function, model, self.grid_size, self.min_vals,
                                                          self.max_vals, self.inds, params, x, death_ind)
                    move_logprob = reverse_logprob - p_params[-1]
                else:
                    # requires multiple moves
                    move_logprob = np.NINF
        else:
            # requires multiple moves
            move_logprob = np.NINF

        return move_logprob

    def eval_all(self, x, params, y, p_params):
        # evaluate all possible
        dim_x = x.value
        dim_y = y.value

        evals = []
        if (dim_y == dim_x + 1):
            # Birth move
            reverse_logprob = 1 / dim_y
            offsets = np.asarray([p_params[-dim_y-1]])  # calculate move probability for grid specified by offsets sampled in p_params
            [proposals, forward_logprobs] = grid_search_logprobs(data_function, model, self.grid_size, self.min_vals, self.max_vals, offsets, self.inds, params[0:(dim_x + 1)], y)
            for proposal, forward_logprob in zip(proposals, forward_logprobs):
                move_logprob = reverse_logprob - forward_logprob
                evals.append((proposal, move_logprob))
        elif (dim_y == dim_x - 1):
            # Death move
            theta0 = params[0:dim_x]
            delta2 = params[dim_x]
            forward_logprob = 1 / dim_y
            for death_ind in range(0, dim_x):
                if dim_y > 0:
                    theta = np.delete(theta0, death_ind)
                else:
                    theta = np.array([])
                reverse_logprob = reverse_grid_search(data_function, model, self.grid_size, self.min_vals,
                                                      self.max_vals, self.inds, params, x, death_ind)
                move_logprob = reverse_logprob - forward_logprob
                evals.append((np.hstack((theta, delta2, params[dim_x + 1:-1])), move_logprob))
        else:
            evals = [(np.asarray([]), np.NINF)]

        return evals

#np.seterr(all='raise')  # DEBUG

# initialise stan model
model = stan_model(stan_model_path, bridgestan_path, cmdstan_path)

# set variables used in the proposal
rj.set_proposal_attributes(spec.SpectrumDimensionTarget(3, 2), model, data_function, continuous_proposal(), "NUTS", 0.5)

# define target
target = rj.ReversibleJumpTarget()
initialProposal = rj.ReversibleJumpInitialProposal(spec.SpectrumDimension, spec.SpectrumDimensionInitialProposal(1))

specSMC = DiscreteVariableSMC(rj.ReversibleJumpVariable, target, initialProposal)

with open(data_path, 'r') as f:
    json_data = (json.load(f))
    for key, value in json_data.items():
        stationary_data.append([key, value])

try:
    SMCSamples = specSMC.sample(10, 10)

except ZeroDivisionError:
    print("SMC sampling failed due to division by zero")
