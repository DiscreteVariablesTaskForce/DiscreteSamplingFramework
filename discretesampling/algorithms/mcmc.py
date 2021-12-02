import random
import math
import copy

class DiscreteVariableMCMC():

    def __init__(self, variableType, target, initialProposal):
        self.variableType = variableType
        self.proposalType = variableType.getProposalType()
        self.initialProposal = initialProposal
        self.target = target
    
    def sample(self, N):
        initialSample = self.initialProposal.sample()
        current = initialSample

        samples = []
        for i in range(N):
            forward_proposal = self.proposalType(current)
            proposed = forward_proposal.sample()

            reverse_proposal = self.proposalType(proposed)
            
            forward_logprob = forward_proposal.eval(proposed)
            reverse_logprob = reverse_proposal.eval(current)

            current_target_logprob = self.target.eval(current)
            proposed_target_logprob = self.target.eval(proposed)
            
            log_acceptance_ratio = proposed_target_logprob - current_target_logprob + reverse_logprob - forward_logprob
            acceptance_probability = min(1, math.exp(log_acceptance_ratio))

            q = random.random()
            #Accept/Reject
            if (q < acceptance_probability):
                current = proposed
            else:
                #Do nothing
                pass
            
            samples.append(copy.deepcopy(current))
        
        return samples