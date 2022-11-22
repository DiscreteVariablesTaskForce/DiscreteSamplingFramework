from ..base import types
from ..base.algorithms import rjmcmc
from scipy.stats import nbinom
import numpy as np
import copy


def set_proposal_attributes(base_DiscreteVariableTarget, stan_model, data_function, continuous_proposal,
                            continuous_update="NUTS", update_probability=0.5):
    # attributes used in proposal (must be set prior to sampling)
    global base_target
    global model
    global data_func
    global cont_proposal
    global cont_update
    global update_prob
    base_target = base_DiscreteVariableTarget
    model = stan_model
    data_func = data_function
    cont_proposal = continuous_proposal
    cont_update = continuous_update
    update_prob = update_probability


# ReversibleJumpVariable inherits from DiscreteVariable
class ReversibleJumpVariable(types.DiscreteVariable):
    def __init__(self, DiscreteVariables, ContinuousVariables):
        self.discrete = DiscreteVariables
        self.continuous = ContinuousVariables

        # need to store the current and proposed momenta, plus NUTS parameters if we are using NUTS
        self.r0 = 0
        self.r1 = 0
        self.NUTS_params = {}

    @classmethod
    def getProposalType(self):
        return ReversibleJumpProposal

    @classmethod
    def getTargetType(self):
        return ReversibleJumpTarget


# ReversibleJumpProposal uses discrete proposals defined by other DiscreteVariableProposal
# The main purpose of this proposal class proposes new continuous parameters based on the discrete proposal
class ReversibleJumpProposal(types.DiscreteVariableProposal):
    def __init__(self, currentReversibleJumpVariable):
        self.currentReversibleJumpVariable = currentReversibleJumpVariable
        self.rjmcmc = rjmcmc.DiscreteVariableRJMCMC(type(currentReversibleJumpVariable.discrete), base_target, model,
                                                    data_func, cont_proposal, cont_update, True, True, False, update_prob)
        # copy over previous NUTS parameters
        if cont_update == "NUTS":
            self.rjmcmc.csampler.NUTS_params = self.currentReversibleJumpVariable.NUTS_params

    @classmethod
    def norm(self, x):
        return self.base_DiscreteVariableProposal.norm(x.discrete)

    @classmethod
    def heuristic(self, x, y):
        # Proposal can be type specified by discrete variables or continuous update
        return self.base_DiscreteVariableProposal.heuristic(x.discrete, y.discrete) or x.discrete == y.discrete

    def sample(self):

        proposedReversibleJumpVariable = copy.deepcopy(self.currentReversibleJumpVariable)

        current_discrete = self.currentReversibleJumpVariable.discrete
        current_continuous = self.currentReversibleJumpVariable.continuous

        [proposed_discrete, proposed_continuous, r0, r1] = self.rjmcmc.propose(current_discrete, current_continuous)
        proposedReversibleJumpVariable.discrete = proposed_discrete
        proposedReversibleJumpVariable.continuous = proposed_continuous

        if cont_update == "NUTS":
            proposedReversibleJumpVariable.r0 = r0
            proposedReversibleJumpVariable.r1 = r1
            proposedReversibleJumpVariable.NUTS_params = self.rjmcmc.csampler.NUTS_params

        return proposedReversibleJumpVariable

    def eval(self, proposedReversibleJumpVariable):
        current_discrete = self.currentReversibleJumpVariable.discrete
        current_continuous = self.currentReversibleJumpVariable.continuous

        proposed_discrete = proposedReversibleJumpVariable.discrete
        proposed_continuous = proposedReversibleJumpVariable.continuous

        r0 = proposedReversibleJumpVariable.r0
        r1 = proposedReversibleJumpVariable.r1

        log_acceptance_ratio = self.rjmcmc.eval(current_discrete, current_continuous, proposed_discrete, proposed_continuous, r0, r1)

        if log_acceptance_ratio > 0:
            log_acceptance_ratio = 0

        return log_acceptance_ratio


class ReversibleJumpTarget(types.DiscreteVariableTarget):

    def eval(self, proposedReversibleJumpVariable):
        proposed_discrete = proposedReversibleJumpVariable.discrete
        proposed_continuous = proposedReversibleJumpVariable.continuous
        proposed_data = data_func(proposed_discrete)
        param_length = model.num_unconstrained_parameters(proposed_data)
        proposed_continuous_target = model.eval(proposed_data, proposed_continuous[0:param_length])[0]
        proposed_discrete_target = base_target.eval(proposed_discrete)
        target = proposed_continuous_target + proposed_discrete_target
        return target


class ReversibleJumpInitialProposal(types.DiscreteVariableInitialProposal):
    def __init__(self, base_DiscreteVariableType, base_DiscreteVariableInitialProposal):
        self.base_type = base_DiscreteVariableType
        self.base_proposal = base_DiscreteVariableInitialProposal

    def sample(self):
        rjmcmc_proposal = rjmcmc.DiscreteVariableRJMCMC(self.base_type, base_target, model,
                                                        data_func, cont_proposal, cont_update, True, True, False, update_prob)
        proposed_discrete = self.base_proposal.sample()
        if hasattr(proposed_discrete.value, "__len__"):
            empty_discrete = self.base_type(np.zeros(proposed_discrete.value.shape()))
        else:
            empty_discrete = self.base_type(0)
        # get initial continuous proposal by performing a birth move from a 0-dimensional model
        proposed_continuous = cont_proposal.sample(empty_discrete, np.array([]), proposed_discrete, np.random.default_rng())

        # do an initial continuous move in case we need to initialise NUTS
        [proposed_continuous, r0, r1] = rjmcmc_proposal.csampler.sample(proposed_continuous, proposed_discrete)

        proposedReversibleJumpVariable = ReversibleJumpVariable(proposed_discrete, proposed_continuous)

        if cont_update == "NUTS":
            proposedReversibleJumpVariable.r0 = r0
            proposedReversibleJumpVariable.r1 = r1
            proposedReversibleJumpVariable.NUTS_params = rjmcmc_proposal.csampler.NUTS_params

        return proposedReversibleJumpVariable

    def eval(self, proposedReversibleJumpVariable):
        proposed_discrete = proposedReversibleJumpVariable.discrete
        if hasattr(proposed_discrete.value, "__len__"):
            empty_discrete = self.base_type(np.zeros(proposed_discrete.value.shape()))
        else:
            empty_discrete = self.base_type(0)
        proposed_continuous = proposedReversibleJumpVariable.continuous
        continuous_logprob = cont_proposal.eval(empty_discrete, np.array([]), proposed_discrete, proposed_continuous)
        discrete_logprob = self.base_proposal.eval(proposed_discrete)
        return continuous_logprob + discrete_logprob
