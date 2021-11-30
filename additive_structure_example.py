import discretesampling.additive_structure as addstruct

# Starting with a discrete set of numbers
initial_set = [[1, 2], [3], [4, 5], [6], [7, 8, 9]]
initialAS = addstruct.AdditiveStructure(initial_set)

forward_proposal = addstruct.AdditiveStructureDistribution(initialAS)
proposedAS = forward_proposal.sample()

forward_probability = forward_proposal.eval(proposedAS)

reverse_proposal = addstruct.AdditiveStructureDistribution(proposedAS)

reverse_probability = reverse_proposal.eval(initialAS)

proposalratio = reverse_probability - forward_probability

