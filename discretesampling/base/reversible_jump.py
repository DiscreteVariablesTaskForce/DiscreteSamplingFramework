from typing import Type
import numpy as np
from scipy.special import logsumexp
from discretesampling.base.algorithms import rjmcmc
from discretesampling.base.random import RNG
from discretesampling.base.types import (
    DiscreteVariable, DiscreteVariableTarget,
    DiscreteVariableProposal, DiscreteVariableInitialProposal
)
from discretesampling.base.algorithms.continuous import ContinuousSampler, NUTS


class ReversibleJumpParameters():
    def __init__(
        self,
        update_probability=0.5,
        continuous_sampler: ContinuousSampler = NUTS(),
        **kwargs
    ):
        self.update_probability = update_probability
        self.continuous_sampler = continuous_sampler
        self.__dict__.update(kwargs)


class ReversibleJumpVariable(DiscreteVariable):
    def __init__(self, discrete: DiscreteVariable, continuous: np.ndarray):
        self.discrete = discrete
        self.continuous = continuous

    def __eq__(self, other):
        return (self.discrete == other.discrete) and (self.continuous == other.continuous)

    @classmethod
    def getProposalType(self):
        return ReversibleJumpProposal

    @classmethod
    def getTargetType(self):
        return ReversibleJumpTarget


# ReversibleJumpProposal uses discrete proposals defined by other DiscreteVariableProposal
# The main purpose of this proposal class proposes new continuous parameters based on the discrete proposal
class ReversibleJumpProposal(DiscreteVariableProposal):
    def __init__(
            self,
            discrete_proposal,
            params: ReversibleJumpParameters = ReversibleJumpParameters()
    ):
        self.params = params
        self.discrete_proposal = discrete_proposal

    @classmethod
    def norm(self, x: ReversibleJumpVariable):
        return self.discrete_proposal.norm(x.discrete)

    @classmethod
    def heuristic(self, x, y):
        # Proposal can be type specified by discrete variables or continuous update
        return self.discrete_proposal.heuristic(x.discrete, y.discrete) or x.discrete == y.discrete

    def propose(self, current_discrete, current_continuous):
        proposed_discrete = current_discrete
        proposed_continuous = current_continuous

        return proposed_discrete, proposed_continuous

    def sample(self, current: ReversibleJumpVariable, rng: RNG = RNG(), target: 'ReversibleJumpTarget' = None):
        [proposed_discrete, proposed_continuous] = self.propose(current.discrete, current.continuous)
        proposed = ReversibleJumpVariable(proposed_discrete, proposed_continuous)

        return proposed

    def eval(self, current, proposed, target: 'ReversibleJumpTarget' = None):
        if current.discrete == proposed.discrete:
            log_acceptance_ratio = self.eval_continuous(current, proposed)
        else:
            log_acceptance_ratio = self.eval_discrete(current, proposed)

        log_acceptance_ratio = min(0, log_acceptance_ratio)
        return log_acceptance_ratio

    def eval_continuous(self, current, proposed):
        # continuous move
        log_acceptance_ratio = (
            self.params.continuous_sampler.eval(current.continuous, current.discrete, proposed.continuous)
            - np.log(self.params.update_probability)
        )
        return log_acceptance_ratio

    def eval_discrete(self, current, proposed):
        # discrete move
        forward_proposal = self.discrete_proposal
        reverse_proposal = self.discrete_proposal
        discrete_forward_logprob = forward_proposal.eval(current.discrete, proposed.discrete)
        discrete_reverse_logprob = reverse_proposal.eval(proposed.discrete, current.discrete)

        continuous_proposal_logprob = self.continuous_proposal.eval(
            current.discrete, current.continuous,
            proposed.discrete, proposed.continuous
        )

        # Setup data for stan model
        current_data = self.data_function(current.discrete)
        proposed_data = self.data_function(proposed.discrete)
        param_length = self.stan_model.num_unconstrained_parameters(current_data)
        p_param_length = self.stan_model.num_unconstrained_parameters(proposed_data)

        # P(theta | discrete_variables)
        current_continuous_target = self.stan_model.eval(current_data, current.continuous[0:param_length])[0]
        proposed_continuous_target = self.stan_model.eval(proposed_data, proposed.continuous[0:p_param_length])[0]

        # P(discrete_variables)
        current_discrete_target = self.discrete_target.eval(current.discrete)
        proposed_discrete_target = self.discrete_target.eval(proposed.discrete)

        jacobian = 0  # math.log(1)

        log_acceptance_ratio = (proposed_continuous_target - current_continuous_target
                                + proposed_discrete_target - current_discrete_target
                                + discrete_reverse_logprob - discrete_forward_logprob
                                + continuous_proposal_logprob + jacobian
                                - np.log(1 - self.params.update_probability))
        return log_acceptance_ratio


class ReversibleJumpTarget(DiscreteVariableTarget):
    def __init__(self, discreteTarget: DiscreteVariableTarget, continuousTarget, data_function):
        self.discreteTarget = discreteTarget
        self.continuousTarget = continuousTarget
        self.data_function = data_function

    def eval(self, x: ReversibleJumpVariable) -> float:
        proposed_data = self.data_function(x.discrete)
        param_length = self.continuousTarget.num_unconstrained_parameters(proposed_data)

        # Evaluate continuous and discrete logprobs
        logprob_continuous, gradient = self.continuousTarget.eval(proposed_data, x.continuous[0:param_length])
        logprob_discrete = self.discreteTarget.eval(x.discrete)

        logprob = logprob_continuous + logprob_discrete
        return logprob


class ReversibleJumpInitialProposal(DiscreteVariableInitialProposal):
    def __init__(
        self,
        base_DiscreteVariableType: Type[DiscreteVariable],
        base_DiscreteVariableInitialProposal,
        reversibleJumpParameters: ReversibleJumpParameters = None
    ):
        self.base_type = base_DiscreteVariableType
        self.base_initial_proposal = base_DiscreteVariableInitialProposal
        self.params = reversibleJumpParameters

    def sample(self, rng: RNG = RNG(), target: ReversibleJumpTarget = None):
        proposed_discrete = self.base_initial_proposal.sample(rng)
        empty_discrete = self.base_type()
        # get initial continuous proposal by performing a birth move from a 0-dimensional model
        proposed_continuous = self.params.continuous_proposal.sample(empty_discrete, np.array([]), proposed_discrete, rng=rng)

        # do an initial continuous move in case we need to initialise NUTS
        [proposed_continuous, r0, r1] = self.params.continuous_sampler.sample(proposed_continuous, proposed_discrete, rng=rng)

        proposed = ReversibleJumpVariable(proposed_discrete, proposed_continuous)

        return proposed

    def eval(self, proposed, target=None):
        proposed_discrete = proposed.discrete
        if hasattr(proposed_discrete.value, "__len__"):
            empty_discrete = self.base_type(np.zeros(proposed_discrete.value.shape()))
        else:
            empty_discrete = self.base_type(0)
        proposed_continuous = proposed.continuous
        continuous_logprob = g_cont_proposal_type().eval(empty_discrete, np.array([]), proposed_discrete, proposed_continuous)
        discrete_logprob = self.base_proposal.eval(proposed_discrete)
        return continuous_logprob + discrete_logprob
