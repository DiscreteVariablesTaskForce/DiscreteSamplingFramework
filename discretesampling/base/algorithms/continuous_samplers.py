import copy
import numpy as np
from scipy.stats import multivariate_normal


class rw():
    # Random walk sampling
    def __init__(self, model, data_function, rng):
        self.stan_model = model
        self.data_function = data_function
        self.rng = rng

    def sample(self, current_continuous, current_discrete):
        current_data = self.data_function(current_discrete)
        param_length = self.stan_model.num_unconstrained_parameters(current_data)
        mu = [0 for i in range(param_length)]
        sigma = np.identity(param_length) * 1
        proposed_continuous = current_continuous + self.rng.multivariate_normal(mu, sigma)

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
        acceptance_probability = min(1, np.exp(log_acceptance_ratio))

        return proposed_continuous, acceptance_probability


class nuts():
    # We need to adapt the NUTS parameters before we sample from a specific model as defined by the discrete parameters.
    # Each time we jump to a new model, initialise NUTS to adapt the mass matrix and step size parameters and then store these
    # for later use.
    def __init__(self, warmup, model, data_function, rng, adapt_delta=0.9):
        self.discrete = []
        self.NUTS_params = {}
        self.current_stepsize = None
        self.warmup = warmup
        self.stan_model = model
        self.data_function = data_function
        self.rng = rng
        self.delta = adapt_delta

    def init_adapt(self, warmup_iters=100):
        self.warmup_iters = warmup_iters  # warmup follows the same adaptation scheme as Stan

    # Perform a single leapfrog step
    def Leapfrog(self, current_data, theta, r, epsilon):
        [L, dtheta] = self.stan_model.eval(current_data, theta)
        r1 = r + epsilon*np.array(dtheta)/2.0
        theta1 = theta + epsilon*r
        [L1, dtheta_list] = self.stan_model.eval(current_data, theta1)
        r1 = r1 + epsilon*np.array(dtheta_list)/2.0

        return theta1, r1, L1, L

    # Recursive part of NUTS algorithm which decides direction of tree growth
    def BuildTree(self, current_data, theta, r, u, v, j, epsilon, treedepth, init_energy, init_potential, M):
        treedepth += 1
        if treedepth > 10:
            print("max tree depth exceeded")
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
        Delta_max = 1000
        n1 = 0
        s1 = 0
        if j == 0:
            # take one leapfrog step
            [theta1, r1, L1] = self.Leapfrog(current_data, theta, r, v*epsilon)[0:3]
            arg = L1 - 0.5 * np.dot(np.linalg.solve(M, r1), r1) + np.log(np.linalg.det(M)) - init_potential
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
            alpha1 = np.exp(arg - init_energy)
            if alpha1 > 1:
                alpha1 = 1
            n_alpha1 = 1
        else:
            # build the left and right subtrees using recursion
            [theta_n, r_n, theta_p, r_p, theta1, n1, s1, alpha1, n_alpha1, depth_exceeded] = self.BuildTree(current_data,
                                                                                                            theta, r, u, v,
                                                                                                            j-1, epsilon,
                                                                                                            treedepth,
                                                                                                            init_energy,
                                                                                                            init_potential, M)
            if depth_exceeded == 1:
                return 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
            if s1 == 1:
                if v < 0:
                    [theta_n, r_n, _, _, theta2, n2, s2, alpha2, n_alpha2, depth_exceeded] = self.BuildTree(current_data,
                                                                                                            theta_n, r_n, u,
                                                                                                            v, j-1, epsilon,
                                                                                                            treedepth,
                                                                                                            init_energy,
                                                                                                            init_potential, M)
                else:
                    [_, _, theta_p, r_p, theta2, n2, s2, alpha2, n_alpha2, depth_exceeded] = self.BuildTree(current_data,
                                                                                                            theta_p, r_p, u,
                                                                                                            v, j-1, epsilon,
                                                                                                            treedepth,
                                                                                                            init_energy,
                                                                                                            init_potential, M)
                if depth_exceeded == 1:
                    return 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
                if n1 != 0 and n2 != 0:
                    u1 = self.rng.uniform(0, 1)
                    if u1 < n2/(n1+n2):
                        theta1 = theta2
                alpha1 = alpha1 + alpha2
                n_alpha1 = n_alpha1 + n_alpha2
                arg = theta_p - theta_n
                I1 = 0
                if np.dot(arg, r_n) >= 0:
                    I1 = 1
                I2 = 0
                if np.dot(arg, r_p) >= 0:
                    I2 = 1
                s1 = s2*I1*I2
                n1 = n1 + n2

        return theta_n, r_n, theta_p, r_p, theta1, n1, s1, alpha1, n_alpha1, 0

    def NUTS(self, current_continuous, current_discrete, M, epsilon):
        current_data = self.data_function(current_discrete)
        param_length = self.stan_model.num_unconstrained_parameters(current_data)

        # calculate likelihood for starting state
        L = self.stan_model.eval(current_data, current_continuous)[0]

        # randomly sample momenta (mass matrix not currently implemented)
        r0 = self.rng.multivariate_normal(np.zeros([param_length]), np.identity(param_length))

        # calculate energy
        init_energy = -0.5 * np.dot(np.linalg.solve(M, r0), r0)
        # more numerically-stable if energy is normalised relative to initial
        # logp value (which then needs to be passed into BuildTree())
        arg = init_energy + np.log(np.linalg.det(M))
        u = self.rng.uniform(0, np.exp(arg))
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
            v_j = self.rng.uniform(-1, 1)
            if v_j < 0:
                [theta_n, r_n, _, _, theta1, n1, s1, alpha, n_alpha, depth_exceeded] = self.BuildTree(current_data, theta_n,
                                                                                                      r_n, u, v_j, j,
                                                                                                      epsilon, treedepth,
                                                                                                      init_energy, L, M)
            else:
                [_, _, theta_p, r_p, theta1, n1, s1, alpha, n_alpha, depth_exceeded] = self.BuildTree(current_data, theta_p,
                                                                                                      r_p, u, v_j, j,
                                                                                                      epsilon, treedepth,
                                                                                                      init_energy, L, M)
            if depth_exceeded == 1:
                # max tree depth exceeded, restart from beginning
                r0 = self.rng.multivariate_normal(np.zeros([param_length]), np.identity(param_length))
                init_energy = -0.5*np.dot(np.linalg.solve(M, r0), r0)
                arg = init_energy + np.log(np.linalg.det(M))
                u = self.rng.uniform(0, np.exp(arg))
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
                    u1 = self.rng.uniform(0, 1)
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

        return p_params, alpha, n_alpha

    def FindReasonableEpsilon(self, current_continuous, current_discrete):
        epsilon = 1
        current_data = self.data_function(current_discrete)
        param_length = self.stan_model.num_unconstrained_parameters(current_data)

        # randomly sample momenta
        r = self.rng.multivariate_normal(np.zeros([param_length]), np.identity(param_length))

        [theta1, r1, L1, L] = self.Leapfrog(current_data, current_continuous, r, epsilon)

        # work out whether we need to increase or decrease step size
        u = 0
        if (L1 - L) > np.log(0.5):
            u = 1
        a = 2*u - 1

        # iteratively adjust step size until the acceptance probability is approximately 0.5
        while a*(L1 - L) > -a*np.log(2):
            epsilon = (2**a)*epsilon
            theta1 = self.Leapfrog(current_data, current_continuous, r, epsilon)[0]
            L1 = self.stan_model.eval(current_data, theta1)[0]

        return epsilon

    def init_NUTS(self, current_continuous, current_discrete):
        current_data = self.data_function(current_discrete)
        param_length = self.stan_model.num_unconstrained_parameters(current_data)

        M = np.identity(param_length)  # currently no mass matrix adaptation (assume identity matrix)

        # Step 1: Get initial estimate for optimal step size
        if self.current_stepsize is None:
            log_epsilon = np.log(self.FindReasonableEpsilon(current_continuous, current_discrete))
        else:
            log_epsilon = np.log(self.current_stepsize)

        # Step 2: Adapt step size using dual averaging
        mu = np.log(10) + log_epsilon
        gamma = 0.05
        t0 = 10
        kappa = 0.75
        H_bar = 0
        log_epsilon_bar = 0
        proposed_continuous = copy.deepcopy(current_continuous)
        for n in range(1, self.warmup_iters):
            proposed_continuous, alpha, n_alpha = self.NUTS(proposed_continuous, current_discrete, M, np.exp(log_epsilon))
            H_bar = (1 - 1/(n + t0))*H_bar + (self.delta - alpha/n_alpha)/(n + t0)
            log_epsilon = mu - np.sqrt(n)*H_bar/gamma
            n_scaled = n**(-kappa)
            log_epsilon_bar = n_scaled*log_epsilon + (1-n_scaled)*log_epsilon_bar
        if hasattr(current_discrete.value, "__iter__"):
            self.NUTS_params[','.join(current_discrete.value)] = (M, np.exp(log_epsilon_bar))
        else:
            self.NUTS_params[str(current_discrete.value)] = (M, np.exp(log_epsilon_bar))

        # save step size for initialising other parameter spaces
        self.current_stepsize = np.exp(log_epsilon_bar)

        return proposed_continuous

    def sample(self, current_continuous, current_discrete):
        if current_discrete.value in self.discrete:
            if hasattr(current_discrete.value, "__iter__"):
                M, epsilon = self.NUTS_params[','.join(current_discrete.value)]
            else:
                M, epsilon = self.NUTS_params[str(current_discrete.value)]
            proposed_continuous = self.NUTS(current_continuous, current_discrete, M, epsilon)[0]
        else:
            proposed_continuous = self.init_NUTS(current_continuous, current_discrete)
            self.discrete.append(current_discrete.value)
        return proposed_continuous
