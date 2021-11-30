import random
import math
import copy

class DiscreteVariableMCMC():

    def __init__(self, variableType, target):
        self.variableType = variableType
        self.proposalType = variableType.getProposalType()
        self.target = target
    
    def sample(self, N, init):
        initialSample = init
        assert type(init) is self.variableType, "init is not of correct type"
        
        current = initialSample

        samples = [current]
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