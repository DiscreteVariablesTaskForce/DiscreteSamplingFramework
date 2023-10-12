import copy
import numpy as np
from scipy.stats import multivariate_normal
from discretesampling.base.algorithms.continuous.base import ContinuousSampler


class RandomWalk(ContinuousSampler):
    def __init__(self, model, data_function, rng):
        self.stan_model = model
        self.data_function = data_function
        self.rng = rng

    def sample(self, current_continuous, current_discrete):
        proposed_continuous = copy.deepcopy(current_continuous)
        current_data = self.data_function(current_discrete)
        param_length = self.stan_model.num_unconstrained_parameters(current_data)
        mu = [0 for i in range(param_length)]
        sigma = np.identity(param_length) * 1
        proposed_continuous[0:param_length] = current_continuous[0:param_length] + self.rng.randomMvNormal(mu, sigma)

        return proposed_continuous, 0, 0  # zeros added for consistency with NUTS method

    def eval(self, current_continuous, current_discrete, proposed_continuous, r0, r1):  # r0 and r1 added for consistency
        current_data = self.data_function(current_discrete)
        param_length = self.stan_model.num_unconstrained_parameters(current_data)
        sigma = np.identity(param_length) * 1

        current_target = self.stan_model.eval(current_data, current_continuous[0:param_length])[0]
        proposed_target = self.stan_model.eval(current_data, proposed_continuous[0:param_length])[0]
        forward_logprob = multivariate_normal.logpdf(proposed_continuous[0:param_length],
                                                     mean=current_continuous[0:param_length], cov=sigma)
        reverse_logprob = multivariate_normal.logpdf(current_continuous[0:param_length],
                                                     mean=proposed_continuous[0:param_length], cov=sigma)
        # Discrete part of target p(discrete_variables) cancels
        log_acceptance_ratio = (proposed_target - current_target
                                + reverse_logprob - forward_logprob)
        log_acceptance_ratio = min(0, log_acceptance_ratio)

        return log_acceptance_ratio
