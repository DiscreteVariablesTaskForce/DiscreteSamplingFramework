import discretesampling.additive_structure as addstruct
import math

# Starting with a discrete set of numbers
initial_set = [[1, 2], [3], [4, 5], [6], [7, 8, 9]]
initialAS = addstruct.AdditiveStructure(initial_set)

data = [] #some data defining the target
target = addstruct.AdditiveStructureTarget(data)

forward_proposal = addstruct.AdditiveStructureProposal(initialAS)
proposedAS = forward_proposal.sample()

forward_logprob = forward_proposal.eval(proposedAS)

reverse_proposal = addstruct.AdditiveStructureProposal(proposedAS)

reverse_logprob = reverse_proposal.eval(initialAS)

current_target_logprob = target.eval(initialAS)
proposed_target_logprob = target.eval(proposedAS)

log_acceptance_ratio = proposed_target_logprob - current_target_logprob + reverse_logprob - forward_logprob
acceptance_probability = min(1, math.exp(log_acceptance_ratio))
