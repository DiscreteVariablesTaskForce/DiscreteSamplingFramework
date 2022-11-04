import math
import copy
import numpy as np
from ...base.random import Random
from ...base.algorithms.continuous_samplers import rw, nuts


class DiscreteVariableRJMCMC():

    def __init__(self, variableType, initialProposal, discrete_target,
                 stan_model, data_function, continuous_proposal, continuous_update,
                 update_probability=0.5):

        self.variableType = variableType
        self.proposalType = variableType.getProposalType()
        self.initialProposal = initialProposal
        self.discrete_target = discrete_target

        self.data_function = data_function

        self.data_function = data_function
        self.continuous_proposal = continuous_proposal
        self.continuous_update = continuous_update
        self.stan_model = stan_model
        self.update_probability = update_probability

    def sample(self, N):

        rng = np.random.default_rng()

        current_discrete = self.initialProposal.sample()
        if hasattr(current_discrete.value, "__len__"):
            empty_discrete = self.variableType(np.zeros(current_discrete.value.shape()))
        else:
            empty_discrete = self.variableType(0)

        # get initial continuous proposal by performing a birth move from a 0-dimensional model
        current_continuous = self.continuous_proposal(empty_discrete, np.array([]), current_discrete, rng)[0]

        # initialise samplers for continuous parameters
        if self.continuous_update == "random_walk":
            csampler = rw(self.stan_model, self.data_function, rng)
        elif self.continuous_update == "NUTS":
            csampler = nuts(1, self.stan_model, self.data_function, rng, 0.9)
            csampler.init_adapt(10)
        else:
            raise NameError("Continuous update type not defined")

        samples = []
        for i in range(N):

            q = Random().eval()
            # Update vs Birth/death
            if (q < self.update_probability):
                # Perform update in continuous space
                if self.continuous_update == "random_walk":
                    [proposed_continuous, acceptance_probability] = csampler.sample(current_continuous, current_discrete)
                elif self.continuous_update == "NUTS":
                    proposed_continuous = csampler.sample(current_continuous, current_discrete)
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
                                                                                              proposed_discrete,
                                                                                              rng)

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
