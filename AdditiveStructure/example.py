import numpy as np
from split_merge import *

# Starting with a discrete set of numbers
initial_set = np.array([[1, 2], [3], [4, 5], [6], [7, 8, 9]])

# We want to either split one subset or merge two
sp = SplitMergeMove()
new_set, prob, rev = sp.propose(discrete_set=initial_set)  # the new set, the log probability and reverse
print("New subset", new_set)

log_prob = sp.probability_merge(frac=1, num_set=2)      # log probability of merging two sets
