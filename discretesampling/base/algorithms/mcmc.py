import math
import copy
from tqdm.auto import tqdm
from discretesampling.base.random import RNG


class DiscreteVariableMCMC():

    def __init__(self, variableType, target, initialProposal, proposal=None):
        self.variableType = variableType
        self.proposalType = variableType.getProposalType()
        self.proposal = proposal
        if proposal is None:
            self.proposal = self.proposalType()
        self.initialProposal = initialProposal
        self.target = target

    def sample(self, N, seed=0, verbose=True, callback=None, keep_samples=False, mixed_in=1000):
        rng = RNG(seed)
        initialSample = self.initialProposal.sample(rng)
        current = initialSample

        samples = []
        # Diagnostics: the proportion of proposals accepted, over the whole run
        # and over the last mixed_in iterations. Recorded rather than returned so
        # the return type is unchanged.
        accepted = []

        display_progress_bar = verbose
        progress_bar = tqdm(total=N, desc="MCMC sampling", disable=not display_progress_bar)

        for i in range(N):
            forward_proposal = self.proposal
            proposed = forward_proposal.sample(current, rng=rng)

            reverse_proposal = self.proposal

            forward_logprob = forward_proposal.eval(current, proposed)
            reverse_logprob = reverse_proposal.eval(proposed, current)

            current_target_logprob = self.target.eval(current)
            proposed_target_logprob = self.target.eval(proposed)

            log_acceptance_ratio = proposed_target_logprob -\
                current_target_logprob + reverse_logprob - forward_logprob
            if log_acceptance_ratio > 0:
                log_acceptance_ratio = 0
            acceptance_probability = min(1, math.exp(log_acceptance_ratio))

            q = rng.random()
            # Accept/Reject
            was_accepted = q < acceptance_probability
            if was_accepted:
                current = proposed
                accepted.append(True)
            else:
                # Do nothing
                accepted.append(False)

            if callback is not None:
                callback(i, current, was_accepted)
            if keep_samples:
                samples.append(copy.copy(current))
            progress_bar.update(1)

        progress_bar.close()
        self.acceptance_rate = sum(accepted) / N if N else float('nan')
        tail = accepted[-mixed_in:]
        self.tail_acceptance_rate = sum(tail) / len(tail) if tail else float('nan')
        return samples
