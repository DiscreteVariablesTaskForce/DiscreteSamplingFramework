import math
import copy
import numpy as np
from scipy.stats import multivariate_normal
from ...base.random import Random
from ...base.stan_model import stan_model
from ...base.util import u2c, c2u_array


class DiscreteVariableRJMCMC():

    def __init__(self, variableType, initialProposal, discrete_target,
                 stan_model_path, bridgestan_path, cmdstan_path,
                 data_function, continuous_proposal, continuous_update,
                 update_probability=0.5):

        self.variableType = variableType
        self.proposalType = variableType.getProposalType()
        self.initialProposal = initialProposal
        self.discrete_target = discrete_target

        self.stan_model_path = stan_model_path
        self.bridgestan_path = bridgestan_path
        self.cmdstan_path = cmdstan_path

        self.data_function = data_function

        self.data_function = data_function
        self.continuous_proposal = continuous_proposal
        self.continuous_update = continuous_update
        self.stan_model = None
        self.update_probability = update_probability

    def init_stan_model(self):
        self.stan_model = stan_model(self.stan_model_path, self.bridgestan_path, self.cmdstan_path)

    def random_walk(self, current_continuous, current_discrete, rng):
        current_data = self.data_function(current_discrete)
        param_length = self.stan_model.num_unconstrained_parameters(current_data)
        mu = [0 for i in range(param_length)]
        sigma = np.identity(param_length) * 1
        proposed_continuous = current_continuous + rng.multivariate_normal(mu, sigma)

        current_data = self.data_function(current_discrete)

        current_target = self.stan_model.eval(current_data, current_continuous)[0]
        proposed_target = self.stan_model.eval(current_data, proposed_continuous)[0]
        forward_logprob = multivariate_normal.logpdf(proposed_continuous, mean=current_continuous, cov=sigma)
        reverse_logprob = multivariate_normal.logpdf(current_continuous, mean=proposed_continuous, cov=sigma)

        # Discrete part of target p(discrete_variables) cancels
        log_acceptance_ratio = (proposed_target - current_target
                                + reverse_logprob - forward_logprob)
        if log_acceptance_ratio > 0:
            log_acceptance_ratio = 0
        acceptance_probability = min(1, math.exp(log_acceptance_ratio))

        return proposed_continuous, acceptance_probability

    # Perform a single leapfrog step
    def Leapfrog(self, current_data, theta, r, epsilon):
        dtheta = np.array(self.stan_model.eval(current_data, theta)[1])
        r1 = r + epsilon*dtheta/2.0
        theta1 = theta + epsilon*r
        [L1, dtheta_list] = self.stan_model.eval(current_data, theta1)
        r1 = r1 + epsilon*np.array(dtheta_list)/2.0

        return theta1, r1, L1

    # Recursive part of NUTS algorithm which decides direction of tree growth
    def BuildTree(self, current_data, theta, r, u, v, j, epsilon, treedepth, param_length, rng):
        treedepth += 1
        if treedepth > 15:
            print("max tree depth exceeded")
            return 0, 0, 0, 0, 0, 0, 0, 1
        Delta_max = 1000
        n1 = 0
        s1 = 0
        if j == 0:
            # take one leapfrog step
            [theta1, r1, L1] = self.Leapfrog(current_data, theta, r, v*epsilon)
            arg = L1-0.5*np.dot(r1, r1)
            if u <= np.exp(arg):
                n1 = 1
            if u == 0:
                # stop log(0) error
                s1 = 1
            else:
                if arg > (np.log(u) - Delta_max):
                    s1 = 1
            theta_n = theta1
            r_n = r1
            theta_p = theta1
            r_p = r1
        else:
            # build the left and right subtrees using recursion
            [theta_n, r_n, theta_p, r_p, theta1, n1, s1, depth_exceeded] = self.BuildTree(current_data, theta, r, u, v, j-1,
                                                                                          epsilon, treedepth, param_length,
                                                                                          rng)
            if depth_exceeded == 1:
                return 0, 0, 0, 0, 0, 0, 0, 1
            if s1 == 1:
                if v < 0:
                    [theta_n, r_n, _, _, theta2, n2, s2, depth_exceeded] = self.BuildTree(current_data, theta_n, r_n, u, v,
                                                                                          j-1, epsilon, treedepth,
                                                                                          param_length, rng)
                else:
                    [_, _, theta_p, r_p, theta2, n2, s2, depth_exceeded] = self.BuildTree(current_data, theta_p, r_p, u, v,
                                                                                          j-1, epsilon, treedepth,
                                                                                          param_length, rng)
                if depth_exceeded == 1:
                    return 0, 0, 0, 0, 0, 0, 0, 1
                if n1 != 0 and n2 != 0:
                    u1 = rng.uniform(0, 1)
                    if u1 < n2/(n1+n2):
                        theta1 = theta2
                arg = theta_p - theta_n
                I1 = 0
                if np.dot(arg, r_n) >= 0:
                    I1 = 1
                I2 = 0
                if np.dot(arg, r_p) >= 0:
                    I2 = 1
                s1 = s2*I1*I2
                n1 = n1 + n2

        return theta_n, r_n, theta_p, r_p, theta1, n1, s1, 0

    def NUTS(self, current_continuous, current_discrete, rng):
        current_data = self.data_function(current_discrete)
        param_length = self.stan_model.num_unconstrained_parameters(current_data)

        epsilon = 0.1  # step size is currently hard-coded (needs adaptation)

        # calculate likelihood for starting state
        L = self.stan_model.eval(current_data, current_continuous)[0]

        # randomly sample momenta (mass matrix not currently implemented)
        r0 = rng.multivariate_normal(np.zeros([param_length]), np.identity(param_length))

        arg = L - np.dot(r0, r0)/2
        u = rng.uniform(0, np.exp(arg))
        theta_n = copy.deepcopy(current_continuous)
        theta_p = copy.deepcopy(current_continuous)
        p_params = copy.deepcopy(current_continuous)
        r_n = r0
        r_p = r0
        j = 0
        s = 1
        n = 1
        treedepth = 0

        # start building tree
        while s == 1:
            v_j = rng.uniform(-1, 1)
            if v_j < 0:
                [theta_n, r_n, _, _, theta1, n1, s1, depth_exceeded] = self.BuildTree(current_data, theta_n, r_n, u, v_j, j,
                                                                                      epsilon, treedepth, param_length, rng)
            else:
                [_, _, theta_p, r_p, theta1, n1, s1, depth_exceeded] = self.BuildTree(current_data, theta_p, r_p, u, v_j, j,
                                                                                      epsilon, treedepth, param_length, rng)
            if depth_exceeded == 1:
                # max tree depth exceeded, restart from beginning
                r0 = rng.multivariate_normal(np.zeros([param_length]), np.identity(param_length))
                arg = L - np.dot(r0, r0)/2
                u = rng.uniform(0, np.exp(arg))
                theta_n = copy.deepcopy(current_continuous)
                theta_p = copy.deepcopy(current_continuous)
                p_params = copy.deepcopy(current_continuous)
                r_n = r0
                r_p = r0
                j = 0
                s = 1
                n = 1
                treedepth = 0
            else:
                if s1 == 1:
                    u1 = rng.uniform(0, 1)
                    if n1/n > u1:
                        p_params = theta1
                n += n1
                u1 = 0
                if np.dot((theta_p - theta_n), r_n) >= 0:
                    u1 = 1
                u2 = 0
                if np.dot((theta_p - theta_n), r_p) >= 0:
                    u2 = 1
                s = s1*u1*u2
                j += 1

        return p_params

    def sample(self, N):

        if self.stan_model is None:
            self.init_stan_model()

        rng = np.random.default_rng()

        initialSample = self.initialProposal.sample()
        current_discrete = initialSample
        current_data = self.data_function(current_discrete)
        param_length = self.stan_model.num_unconstrained_parameters(current_data)

        mu = [0 for i in range(param_length)]
        sigma = np.identity(param_length) * 10
        initialSample_continuous = rng.multivariate_normal(mu, sigma)
        current_continuous = initialSample_continuous

        samples = []
        for i in range(N):

            q = Random().eval()
            # Update vs Birth/death
            if (q < self.update_probability):
                # Perform update in continuous space
                if self.continuous_update == "random_walk":
                    [proposed_continuous, acceptance_probability] = self.random_walk(current_continuous, current_discrete, rng)
                elif self.continuous_update == "NUTS":
                    proposed_continuous = self.NUTS(current_continuous, current_discrete, rng)
                    acceptance_probability = 1
                else:
                    raise NameError("Continuous update type not defined")

                q = Random().eval()
                if (q < acceptance_probability):
                    current_continuous = proposed_continuous

            else:
                # Perform discrete update
                forward_proposal = self.proposalType(current_discrete)
                proposed_discrete = forward_proposal.sample()
                reverse_proposal = self.proposalType(proposed_discrete)
                discrete_forward_logprob = forward_proposal.eval(proposed_discrete)
                discrete_reverse_logprob = reverse_proposal.eval(current_discrete)

                # Birth/death continuous dimensions
                # It would probably be better if this was called as above, with separate eval() and sample() functions.
                # However, the random choices made in the forward move would need to be passed to calculate the probability of
                # the reverse move (e.g. random selection for birth / death). This information might be better stored in the
                # discrete variables.
                [proposed_continuous, continuous_proposal_logprob] = self.continuous_proposal(current_discrete,
                                                                                              current_continuous,
                                                                                              proposed_discrete)

                # Setup data for stan model
                current_data = self.data_function(current_discrete)
                proposed_data = self.data_function(proposed_discrete)

                # P(theta | discrete_variables)
                current_continuous_target = self.stan_model.eval(current_data, current_continuous)[0]
                proposed_continuous_target = self.stan_model.eval(proposed_data, proposed_continuous)[0]

                # P(discrete_variables)
                current_discrete_target = self.discrete_target.eval(current_discrete)
                proposed_discrete_target = self.discrete_target.eval(proposed_discrete)

                jacobian = 0  # math.log(1)

                log_acceptance_ratio = (proposed_continuous_target - current_continuous_target
                                        + proposed_discrete_target - current_discrete_target
                                        + discrete_reverse_logprob - discrete_forward_logprob
                                        + continuous_proposal_logprob + jacobian)
                if log_acceptance_ratio > 0:
                    log_acceptance_ratio = 0
                acceptance_probability = min(1, math.exp(log_acceptance_ratio))

                q = Random().eval()
                # Accept/Reject
                if (q < acceptance_probability):
                    current_discrete = proposed_discrete
                    current_continuous = proposed_continuous
            print("Iteration {}, params = {}".format(i, current_continuous))
            samples.append([copy.deepcopy(current_discrete), current_continuous])

        return samples

    def forward_grid_search(self, grid_size, min_vals, max_vals, inds, current_continuous, proposed_discrete):
        # input:
        # grid_size          - number of points across which to sample each new parameter
        # min_vals           - list of minimum values of each constrained parameter in order that they appear in
        #                      current_continuous
        # max_vals           - list of maximum values of each constrained parameter in order that they appear in
        #                      current_continuous
        # inds               - indices of parameters to sample (starting with 0)
        # current_continuous - current unconstrained parameters (including those with unchanging dimensions)
        # proposed_discrete  - proposed discrete parameter (assumed to be 1 parameter controlling birth / death moves)

        dim = proposed_discrete.value

        proposed_data = self.data_function(proposed_discrete)

        num_params = len(min_vals)
        total_params = len(current_continuous)
        num_static_params = total_params - (dim-1)*num_params
        num_grid_points = grid_size**num_params

        # define the grid axes across each dimension
        grid_axes = []
        for n in range(0, num_params):
            step = (max_vals[n] - min_vals[n]) / (grid_size + 1)
            offset = step*Random().eval()  # random offset so that any point in the parameter space can be sampled
            grid_axes.append(np.linspace(offset, max_vals[n] - step + offset, grid_size))

        unconstrained_vals = c2u_array(grid_axes, min_vals, max_vals)
        # transform the grid to the unconstrained space
        grid = np.meshgrid(*unconstrained_vals)
        concatenated_grid = []
        for grid_points in grid:
            concatenated_grid.append(np.reshape(grid_points, num_grid_points))

        grid_points = np.asarray(concatenated_grid)

        params_temp = copy.deepcopy(current_continuous)
        proposed_continuous = np.empty([dim*num_params + num_static_params])
        lp_vals = np.empty([num_grid_points])
        for n in range(0, num_grid_points):
            grid_point = grid_points[:, n]
            param_ind1 = 0
            param_ind2 = 0
            for param_num in range(0, num_params):
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
                proposed_continuous[param_ind2:(param_ind2+dim-1)] = params_temp[param_ind1:(param_ind1+dim-1)]
                proposed_continuous[param_ind2+dim-1] = grid_point[param_num]
                param_ind1 += dim - 1
                param_ind2 += dim  # assuming that we're only adding one to the dimension
            if param_ind1 == (total_params - 1):
                proposed_continuous[param_ind2] = params_temp[param_ind1]
            else:
                proposed_continuous[param_ind2:-1] = params_temp[param_ind1:-1]
            lp_vals[n] = self.stan_model.eval(proposed_data, proposed_continuous)[0]

        # randomly sample grid value using the distribution of log_prob values at each point in the grid
        max_lp = np.max(lp_vals)  # need to normalise log_prob values in order to do sum over prob
        p_vals = np.exp(lp_vals - max_lp)  # need to ensure this never underflows / overflows
        probs = p_vals / np.sum(p_vals)
        samp = np.random.choice(range(0, grid_size), p=probs)
        param_ind1 = 0
        param_ind2 = 0
        for param_num in range(0, num_params):
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
            proposed_continuous[param_ind2:(param_ind2+dim-2)] = params_temp[param_ind1:(param_ind1+dim-2)]
            proposed_continuous[param_ind2 + dim - 1] = grid_points[param_num, samp]
            param_ind1 += dim - 1
            param_ind2 += dim
        proposed_continuous[param_ind2:-1] = params_temp[param_ind1:-1]

        return proposed_continuous, np.log(probs[samp])

    def reverse_grid_search(self, grid_size, min_vals, max_vals, inds, current_continuous, current_discrete, death_ind):
        # input:
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

        current_data = self.data_function(current_discrete)

        num_params = len(min_vals)
        num_grid_points = grid_size**num_params

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
            lp_vals[n] = self.stan_model.eval(current_data, proposed_continuous)[0]

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