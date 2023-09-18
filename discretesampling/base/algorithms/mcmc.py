import math
import copy
from discretesampling.base.random import RNG
from discretesampling.base.types import DiscreteVariable, DiscreteVariableInitialProposal, DiscreteVariableTarget


class DiscreteVariableMCMC():
    """Implementation of MCMC sampler for discrete variables.
    """

    def __init__(
        self,
        variableType: DiscreteVariable,
        target: DiscreteVariableTarget,
        initialProposal: DiscreteVariableInitialProposal
    ):
        """Constructor method

        Parameters
        ----------
        variableType : DiscreteVariable
            Type of variables
        target : DiscreteVariableTarget
            Target (or posterior) distribution to sample from.
        initialProposal : DiscreteVariableInitialProposal
            Proposal distribution of initial samples.
        """

        self.variableType = variableType
        self.proposalType = variableType.getProposalType()
        self.initialProposal = initialProposal
        self.target = target

    def sample(self, N: int, seed: int = 0) -> list[DiscreteVariable]:
        """Generate samples from the MCMC sampler.

        Parameters
        ----------
        N : int
            Number of iterations to run.
        seed : int, optional
            Random seed, by default 0

        Returns
        -------
        list[DiscreteVariable]
            Generated MCMC samples.
        """
        rng = RNG(seed)
        initialSample = self.initialProposal.sample(rng)
        current = initialSample

        samples = []
        for i in range(N):
            forward_proposal = self.proposalType(current, rng)
            proposed = forward_proposal.sample()

            reverse_proposal = self.proposalType(proposed, rng)

            forward_logprob = forward_proposal.eval(proposed)
            reverse_logprob = reverse_proposal.eval(current)

            current_target_logprob = self.target.eval(current)
            proposed_target_logprob = self.target.eval(proposed)

            log_acceptance_ratio = proposed_target_logprob -\
                current_target_logprob + reverse_logprob - forward_logprob
            if log_acceptance_ratio > 0:
                log_acceptance_ratio = 0
            acceptance_probability = min(1, math.exp(log_acceptance_ratio))

            q = rng.random()
            # Accept/Reject
            if (q < acceptance_probability):
                current = proposed
            else:
                # Do nothing
                pass

            samples.append(copy.copy(current))

        return samples
