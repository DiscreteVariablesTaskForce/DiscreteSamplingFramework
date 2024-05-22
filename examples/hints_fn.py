from numba import jit
import math
import numpy as np
import time
from numpy.random import seed, randn, rand, randint, shuffle
#from line_profiler import LineProfiler

from functools import lru_cache # prefer diskcache because it can memoize unhashable types

# diskcache is great at memoizing but slow for fine grain calls
#from diskcache import Cache
#cache = Cache(size_limit=int(1e9)) #1GB
#cache.stats(enable=True)



# template for user function
# cache support at this level has been removed... user should implement
# this avoids memoizing a class member
class UserFn:
    def __init__(self, proposal, additive = True):
        self.additive = additive # user can override
        self.counter = 0 # keeps track of term evaluations
        self.total_counter = 0 # this includes cached ones
        self.proposal = proposal # user provides a proposal function
    #
    # where should the chain start
    def sample_initial_state(self):
        pass # user must implement       
    #
    # the low level evaluation
    # with_gradient is for systems like pytorch that can hang a gradient onto the quantities returned
    def evaluate(self, state, term_index, with_gradient = False):
        pass # user must implement (term in additive structure)
        # user should increment self.counter for cache misses
    #
    # return sum over scenarios of log density term
    # this can be computed in parallel across scenarios, but that would restrict cache benefit
    # so better to use minibatch at the single scenario level (self.evaluate)
    # state could be a numpy array, or a complex structure such as a pyTorch model (state dict)
    def __call__(self, state, scenarios, with_gradient = False):
        n = len(scenarios)
        sum_f  = sum([self.evaluate(state, term, with_gradient) for term in scenarios])
        sum_f += self.evaluate_regularisation(state, with_gradient) * n # OPTIONAL, MUST scale with number of scenarios
        self.total_counter += n # lru cache ignores this side effect
        return(sum_f if self.additive else sum_f/n)
        #            
    def evaluate_regularisation(self, state, with_gradient = False):
        return(0.0) # none by default, user can override
