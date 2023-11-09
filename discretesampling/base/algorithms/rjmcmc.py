import math
import copy
import numpy as np
from scipy.special import logsumexp
from discretesampling.base.random import RNG
from discretesampling.base.algorithms.continuous.NUTS import NUTS
from discretesampling.base.algorithms.continuous.RandomWalk import RandomWalk


class DiscreteVariableRJMCMC():

    def __init__(self, variableType, discrete_target,
                 stan_model, data_function, continuous_proposal_type, continuous_update, accept_reject=False,
                 always_update=False, do_warmup=True, update_probability=0.5, initialProposal=None, warmup_iters=100):

        self.variableType = variableType
        self.proposalType = variableType.getProposalType()
        self.initialProposal = initialProposal
        self.discrete_target = discrete_target

        self.data_function = data_function

        self.data_function = data_function
        self.continuous_proposal_type = continuous_proposal_type
        self.continuous_update = continuous_update
        self.stan_model = stan_model
        self.accept_reject = accept_reject
        self.always_update = always_update
        self.update_probability = update_probability

        # initialise samplers for continuous parameters
        if self.continuous_update is RandomWalk:
            self.csampler = RandomWalk(self.stan_model, self.data_function)
        elif self.continuous_update is NUTS:
            self.csampler = NUTS(self.stan_model, self.data_function, update_probability=0.9,
                                 warmup_iters=warmup_iters, do_warmup=do_warmup)
        else:
            raise NameError("Continuous update type not defined")

    def propose(self, current_discrete, current_continuous, rng):
        proposed_discrete = current_discrete
        q = rng.random()
        r0 = 0
        r1 = 0
        # Update vs Birth/death
        if (q < self.update_probability):
            # Perform update in continuous space
            [proposed_continuous, r0, r1] = self.csampler.sample(current_continuous, current_discrete, rng=rng)
        else:
            # Perform discrete update
            forward_proposal = self.proposalType()
            proposed_discrete = forward_proposal.sample(current_discrete, rng=rng)

            # Birth/death continuous dimensions
            # It would probably be better if this was called as above, with separate eval() and sample() functions.
            # However, the random choices made in the forward move would need to be passed to calculate the probability of
            # the reverse move (e.g. random selection for birth / death). This information might be better stored in the
            # discrete variables.
            if self.always_update:
                # need to do this to initialise NUTS (if we don't do this then we can end up in a situation in SMC where we're
                # comparing NUTS proposals between two starting points without sampled momenta / NUTS parameters)
                init_proposed_continuous = self.continuous_proposal_type().sample(current_discrete, current_continuous,
                                                                                  proposed_discrete,
                                                                                  rng=rng)
                [proposed_continuous, r0, r1] = self.csampler.sample(init_proposed_continuous, proposed_discrete, rng=rng)
            else:
                proposed_continuous = self.continuous_proposal.sample(current_discrete, current_continuous, proposed_discrete,
                                                                      rng=rng)

        return proposed_discrete, proposed_continuous, r0, r1

    def sample(self, N, seed=0):
        rng = RNG(seed=seed)

        current_discrete = self.initialProposal.sample(rng=rng)
        self.continuous_proposal = self.continuous_proposal_type()
        if hasattr(current_discrete.value, "__len__"):
            empty_discrete = self.variableType(np.zeros(current_discrete.value.shape()))
        else:
            empty_discrete = self.variableType(0)

        # get initial continuous proposal by performing a birth move from a 0-dimensional model
        current_continuous = self.continuous_proposal.sample(empty_discrete, np.array([]), current_discrete, rng=rng)
        samples = []
        for i in range(N):
            [proposed_discrete, proposed_continuous, r0, r1] = self.propose(current_discrete, current_continuous, rng=rng)
            log_acceptance_ratio = self.eval(current_discrete, current_continuous, proposed_discrete, proposed_continuous,
                                             r0, r1)
            if log_acceptance_ratio > 0:
                log_acceptance_ratio = 0
            acceptance_probability = min(1, math.exp(log_acceptance_ratio))
            q = rng.random()
            # Accept/Reject
            if (q < acceptance_probability):
                current_discrete = proposed_discrete
                current_continuous = proposed_continuous
            print("Iteration {}, params = {}".format(i, current_continuous))
            samples.append([copy.copy(current_discrete), current_continuous])

        return samples

    def eval(self, current_discrete, current_continuous, proposed_discrete, proposed_continuous, r0, r1):
        if current_discrete == proposed_discrete:
            # continuous parameter update move
            if self.continuous_update == NUTS and self.accept_reject is False:
                log_acceptance_ratio = 0
            else:
                log_acceptance_ratio = self.csampler.eval(current_continuous, current_discrete, proposed_continuous, r0, r1) \
                    - np.log(self.update_probability)
        else:
            # discrete move
            forward_proposal = self.proposalType()
            reverse_proposal = self.proposalType()
            discrete_forward_logprob = forward_proposal.eval(current_discrete, proposed_discrete)
            discrete_reverse_logprob = reverse_proposal.eval(proposed_discrete, current_discrete)

            if self.always_update and not (self.continuous_update == NUTS and self.accept_reject is False):
                # we need to sum over all of the possible combinations of birth / death moves + updates to get to
                # proposed_continuous
                continuous_proposals = self.continuous_proposal_type().eval_all(current_discrete, current_continuous,
                                                                                proposed_discrete, proposed_continuous)
                continuous_proposal_logprobs = []
                for continuous_proposal, continuous_proposal_logprob in continuous_proposals:
                    nuts_logprob = self.csampler.eval(continuous_proposal, proposed_discrete, proposed_continuous, r0, r1)
                    continuous_proposal_logprobs.append(continuous_proposal_logprob + nuts_logprob)
                continuous_proposal_logprob = logsumexp(continuous_proposal_logprobs)
            else:
                continuous_proposal_logprob = self.continuous_proposal.eval(current_discrete, current_continuous,
                                                                            proposed_discrete, proposed_continuous)

            # Setup data for stan model
            current_data = self.data_function(current_discrete)
            proposed_data = self.data_function(proposed_discrete)
            param_length = self.stan_model.num_unconstrained_parameters(current_data)
            p_param_length = self.stan_model.num_unconstrained_parameters(proposed_data)

            # P(theta | discrete_variables)
            current_continuous_target = self.stan_model.eval(current_data, current_continuous[0:param_length])[0]
            proposed_continuous_target = self.stan_model.eval(proposed_data, proposed_continuous[0:p_param_length])[0]

            # P(discrete_variables)
            current_discrete_target = self.discrete_target.eval(current_discrete)
            proposed_discrete_target = self.discrete_target.eval(proposed_discrete)

            jacobian = 0  # math.log(1)

            log_acceptance_ratio = (proposed_continuous_target - current_continuous_target
                                    + proposed_discrete_target - current_discrete_target
                                    + discrete_reverse_logprob - discrete_forward_logprob
                                    + continuous_proposal_logprob + jacobian - np.log(1 - self.update_probability))

        return log_acceptance_ratio
