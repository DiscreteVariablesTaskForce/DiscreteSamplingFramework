import copy
import numpy as np
from ...base.random import Random
from ...base.util import u2c, c2u_array


def forward_grid_search(data_function, stan_model, grid_size, min_vals, max_vals, inds, current_continuous, proposed_discrete):
    # input:
    # data_function      - function returning data passed to Stan model for given discrete parameters
    # stan_model         - initialised stan_model object
    # grid_size          - number of points across which to sample each new parameter
    # min_vals           - list of minimum values of each constrained parameter in order that they appear in
    #                      current_continuous
    # max_vals           - list of maximum values of each constrained parameter in order that they appear in
    #                      current_continuous
    # inds               - indices of parameters to sample (starting with 0)
    # current_continuous - current unconstrained parameters (including those with unchanging dimensions)
    # proposed_discrete  - proposed discrete parameter (assumed to be 1 parameter controlling birth / death moves)

    dim = proposed_discrete.value

    proposed_data = data_function(proposed_discrete)

    num_params = len(min_vals)
    total_params = len(current_continuous)
    num_static_params = total_params - (dim - 1) * num_params
    if num_static_params < 0:
        num_static_params = 0
    num_grid_points = grid_size ** num_params

    # define the grid axes across each dimension
    grid_axes = []
    for n in range(0, num_params):
        step = (max_vals[n] - min_vals[n]) / (grid_size + 1)
        offset = step * Random().eval()  # random offset so that any point in the parameter space can be sampled
        grid_axes.append(np.linspace(offset, max_vals[n] - step + offset, grid_size))

    unconstrained_vals = c2u_array(grid_axes, min_vals, max_vals)
    # transform the grid to the unconstrained space
    grid = np.meshgrid(*unconstrained_vals)
    concatenated_grid = []
    for grid_points in grid:
        concatenated_grid.append(np.reshape(grid_points, num_grid_points))

    grid_points = np.asarray(concatenated_grid)

    params_temp = copy.deepcopy(current_continuous)
    proposed_continuous = np.empty([dim * num_params + num_static_params])
    lp_vals = np.empty([num_grid_points])
    for n in range(0, num_grid_points):
        grid_point = grid_points[:, n]
        param_ind1 = 0
        param_ind2 = 0
        for param_num in range(0, num_params):
            if num_static_params != 0:
                if param_num == 0:
                    param_ind1 += inds[param_num]
                    param_ind2 += inds[param_num]
                    proposed_continuous[0:param_ind2 + 1] = params_temp[0:param_ind1 + 1]
                else:
                    old_param_ind1 = copy.deepcopy(param_ind1)
                    old_param_ind2 = copy.deepcopy(param_ind2)
                    param_ind1 += inds[param_num] - inds[param_num - 1]
                    param_ind2 += inds[param_num] - inds[param_num - 1]
                    proposed_continuous[old_param_ind2:param_ind2 + 1] = params_temp[old_param_ind1:param_ind1 + 1]
                proposed_continuous[param_ind2:(param_ind2 + dim - 1)] = params_temp[param_ind1:(param_ind1 + dim - 1)]
            proposed_continuous[param_ind2 + dim - 1] = grid_point[param_num]
            param_ind1 += dim - 1
            param_ind2 += dim  # assuming that we're only adding one to the dimension
        if param_ind1 == (total_params - 1):
            proposed_continuous[param_ind2] = params_temp[param_ind1]
        else:
            proposed_continuous[param_ind2:-1] = params_temp[param_ind1:-1]
        lp_vals[n] = stan_model.eval(proposed_data, proposed_continuous)[0]

    # randomly sample grid value using the distribution of log_prob values at each point in the grid
    max_lp = np.max(lp_vals)  # need to normalise log_prob values in order to do sum over prob
    p_vals = np.exp(lp_vals - max_lp)  # need to ensure this never underflows / overflows
    probs = p_vals / np.sum(p_vals)
    samp = np.random.choice(range(0, grid_size), p=probs)
    param_ind1 = 0
    param_ind2 = 0
    for param_num in range(0, num_params):
        if num_static_params != 0:
            if param_num == 0:
                param_ind1 += inds[param_num]
                param_ind2 += inds[param_num]
                proposed_continuous[0:param_ind2 + 1] = params_temp[0:param_ind1 + 1]
            else:
                old_param_ind1 = copy.deepcopy(param_ind1)
                old_param_ind2 = copy.deepcopy(param_ind2)
                param_ind1 += inds[param_num] - inds[param_num - 1]
                param_ind2 += inds[param_num] - inds[param_num - 1]
                proposed_continuous[old_param_ind2:param_ind2 + 1] = params_temp[old_param_ind1:param_ind1 + 1]
            proposed_continuous[param_ind2:(param_ind2 + dim - 1)] = params_temp[param_ind1:(param_ind1 + dim - 1)]
        proposed_continuous[param_ind2 + dim - 1] = grid_points[param_num, samp]
        param_ind1 += dim - 1
        param_ind2 += dim
    proposed_continuous[param_ind2:-1] = params_temp[param_ind1:-1]

    return proposed_continuous, np.log(probs[samp])


def reverse_grid_search(data_function, stan_model, grid_size, min_vals, max_vals, inds, current_continuous, current_discrete, death_ind):
    # input:
    # data_function      - function returning data passed to Stan model for given discrete parameters
    # stan_model         - initialised stan_model object
    # grid_size          - number of points across which to sample each new parameter
    # min_vals           - list of minimum values of each constrained parameter in order that they appear in
    #                      current_continuous
    # max_vals           - list of maximum values of each constrained parameter in order that they appear in
    #                      current_continuous
    # inds               - indices of parameters to sample (starting with 0)
    # current_continuous - current unconstrained parameters (including those with unchanging dimensions)
    # current_discrete   - current discrete parameter (assumed to be 1 parameter controlling birth / death moves)
    # death_ind          - index of discrete variable proposed for death

    dim = current_discrete.value

    current_data = data_function(current_discrete)

    num_params = len(min_vals)
    num_grid_points = grid_size ** num_params

    # define the grid axes across each dimension
    grid_axes = []
    param_ind = 0
    for param_num in range(0, num_params):
        if param_num == 0:
            param_ind += inds[param_num]
        else:
            param_ind += inds[param_num] - inds[param_num - 1]
        step = (max_vals[param_num] - min_vals[param_num]) / (grid_size + 1)
        offset = np.mod(u2c(current_continuous[param_ind + death_ind], min_vals[param_num], max_vals[param_num]), step)
        grid_axes.append(np.linspace(offset, max_vals[param_num] - step + offset, grid_size))
        param_ind += dim

    unconstrained_vals = c2u_array(grid_axes, min_vals, max_vals)
    # transform the grid to the unconstrained space
    grid = np.meshgrid(*unconstrained_vals)
    concatenated_grid = []
    for grid_points in grid:
        concatenated_grid.append(np.reshape(grid_points, num_grid_points))

    grid_points = np.asarray(concatenated_grid)

    proposed_continuous = copy.deepcopy(current_continuous)
    lp_vals = np.empty([num_grid_points])
    for n in range(0, num_grid_points):
        grid_point = grid_points[:, n]
        param_ind1 = 0
        for param_num in range(0, num_params):
            if param_num == 0:
                param_ind1 += inds[param_num]
            else:
                param_ind1 += inds[param_num] - inds[param_num - 1]
            param_ind2 = param_ind1 + death_ind
            proposed_continuous[param_ind2] = grid_point[param_num]
            param_ind1 += dim - 1
        lp_vals[n] = stan_model.eval(current_data, proposed_continuous)[0]

    # calculate probability of choosing parameter values from specified grid
    max_lp = np.max(lp_vals)  # need to normalise log_prob values in order to do sum over prob
    p_vals = np.exp(lp_vals - max_lp)  # need to ensure this never underflows / overflows
    probs = p_vals / np.sum(p_vals)
    grid_point = np.empty([num_params])
    for param_num in range(0, num_params):
        if param_num == 0:
            param_ind += inds[param_num]
        else:
            param_ind += inds[param_num] - inds[param_num - 1]
        grid_point[param_num] = current_continuous[param_ind]
        param_ind += dim - 1
    ind = (np.abs(grid_points - grid_point)).argmin()
    return np.log(probs[ind])
