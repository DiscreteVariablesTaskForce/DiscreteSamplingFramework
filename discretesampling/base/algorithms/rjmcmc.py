import math
import copy
import numpy as np
from scipy.stats import multivariate_normal
from ...base.random import Random
from ...base.stan_model import stan_model

class DiscreteVariableRJMCMC():

    def __init__(self, variableType, initialProposal, discrete_target,
                 stan_model_path, redding_stan_path,
                 data_function, continuous_dim_function,
                 transformation_function,
                 update_probability = 0.5):
        
        self.variableType = variableType
        self.proposalType = variableType.getProposalType()
        self.initialProposal = initialProposal
        self.discrete_target = discrete_target
        
        self.stan_model_path = stan_model_path
        self.redding_stan_path = redding_stan_path

        self.data_function = data_function
        self.continuous_dim_function = continuous_dim_function
        self.transformation_function = transformation_function
        self.stan_model = None
        self.update_probability = update_probability


    def init_stan_model(self):
        self.stan_model = stan_model(self.stan_model_path, self.redding_stan_path)
        self.stan_model.compile()

    def sample(self, N):

        if self.stan_model is None:
            self.init_stan_model()

        rng = np.random.default_rng()

        initialSample = self.initialProposal.sample()
        current_discrete = initialSample
        param_length = self.continuous_dim_function(current_discrete)

        mu = [0 for i in range(param_length)]
        sigma = np.identity(param_length) * 10
        initialSample_continuous = rng.multivariate_normal(mu,sigma)
        current_continuous = initialSample_continuous

        samples = []
        for i in range(N):

            q = Random().eval()
            # Update vs Birth/death
            if (q < self.update_probability):
                #Perform update in continuous space
                param_length = self.continuous_dim_function(current_discrete)
                mu = [0 for i in range(param_length)]
                sigma = np.identity(param_length) * 1
                proposed_continuous = current_continuous + rng.multivariate_normal(mu,sigma)

                current_data = self.data_function(current_discrete)

                current_target = self.stan_model.eval(current_data, current_continuous)
                proposed_target = self.stan_model.eval(current_data, proposed_continuous)
                forward_logprob = multivariate_normal.logpdf(proposed_continuous, mean = current_continuous, cov = sigma)
                reverse_logprob = multivariate_normal.logpdf(current_continuous, mean = proposed_continuous, cov = sigma)

                #Discrete part of target p(discrete_variables) cancels
                log_acceptance_ratio = (proposed_target - current_target
                                        + reverse_logprob - forward_logprob)
                if log_acceptance_ratio > 0:
                    log_acceptance_ratio = 0
                acceptance_probability = min(1, math.exp(log_acceptance_ratio))

                q = Random().eval()
                if (q < acceptance_probability):
                    current_continuous = proposed_continuous
            
            else:
                #Perform discrete update
                forward_proposal = self.proposalType(current_discrete)
                proposed_discrete = forward_proposal.sample()
                reverse_proposal = self.proposalType(proposed_discrete)
                forward_logprob = forward_proposal.eval(proposed_discrete)
                reverse_logprob = reverse_proposal.eval(current_discrete)

                #Birth/death continuous dimensions
                proposed_continuous = self.transformation_function(current_discrete, current_continuous, proposed_discrete)

                #Setup data for stan model
                current_data = self.data_function(current_discrete)
                proposed_data = self.data_function(proposed_discrete)

                #P(theta | discrete_variables)
                current_continuous_target = self.stan_model.eval(current_data, current_continuous)
                proposed_continuous_target = self.stan_model.eval(proposed_data, proposed_continuous)

                #P(discrete_variables)
                current_discrete_target = self.discrete_target.eval(current_discrete)
                proposed_discrete_target = self.discrete_target.eval(proposed_discrete)

                jacobian = 0 #math.log(1)

                log_acceptance_ratio = (proposed_continuous_target - current_continuous_target 
                                        + proposed_discrete_target - current_discrete_target
                                        + reverse_logprob - forward_logprob
                                        + jacobian)
                if log_acceptance_ratio > 0:
                    log_acceptance_ratio = 0
                acceptance_probability = min(1, math.exp(log_acceptance_ratio))

                q = Random().eval()
                # Accept/Reject
                if (q < acceptance_probability):
                    current_discrete = proposed_discrete
                    current_continuous = proposed_continuous

            samples.append([copy.deepcopy(current_discrete), current_continuous])

        return samples
