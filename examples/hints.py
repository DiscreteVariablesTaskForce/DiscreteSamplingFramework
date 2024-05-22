# the HINTS class
import math
import numpy as np
from numpy.random import seed, randn, rand, shuffle
import random

class HINTS:
    def __init__(self, args, fn, shuffle_as_we_go = False):
        self.args = args
        self.fn = fn # user fn
        self.shuffle_as_we_go = shuffle_as_we_go
        self.levels = args.levels
        self.design = args.design
        self.levels = args.design.shape[0] - 1
        self.Ts = np.flip(np.array([self.args.T + l * self.args.dT for l in range(args.levels + 1)]))
        self.ns = np.cumprod(args.design) # number of scenarios at each level
        self.N = self.ns[args.levels] 
        self.reset()
        # Diagnostics:
        print(self.levels)
        print(self.ns)
        print(self.N)
        print(self.Ts)

    # calculate the terms at this level and index
    def scenarios(self, level, index):
        return([self.fixed_scenarios[self.ns[level] * index + i] for i in range(self.ns[level])])
    
    def reset(self):
        print("RESET")
        self.rejections = np.zeros(self.levels + 1, dtype = int)
        self.acceptances = np.zeros(self.levels + 1, dtype = int)
        self.fixed_scenarios = np.arange(self.ns[self.levels])
        
    def shuffle(self):
        shuffle(self.fixed_scenarios)
    
    def metropolis_accept(self, exp_diff):
        return(True if (exp_diff > 0.0) else (False if (exp_diff < -100.0) else (rand() < math.exp(exp_diff))))
    
    def hints(self, state, level, index = 0, always_accept = False): # main function, requires value of current as input
        if (level == 0): return(self.primitive_move(state, index, always_accept)) # this is only separate so we can override
        scenarios = self.scenarios(level, index)
        correction = 0.0
        current = state # we hold state at all levels in the hierarchy
        branches = list(range(self.design[level]))
        if self.shuffle_as_we_go:
            random.shuffle(branches)
        for b in branches:
            current, delta_correction = self.hints(current, level-1, index * self.design[level] + b) # recursive call
            correction += delta_correction
        # now do composite evaluations AFTER primitive ones, in case primitive ones needed gradients
        vdiff = (self.fn(current, scenarios) - self.fn(state, scenarios))/self.Ts[level] # these are cached evaluations, no side effects
        accept = True if always_accept else self.metropolis_accept(vdiff - correction) # NB second expression will not be evaluated if always accept
        (self.acceptances if accept else self.rejections)[level] += 1
        return((current, vdiff) if accept else (state, 0.0))
    
    # separate out level zero in case we want to override it (e.g. with HMC)
    def primitive_move(self, state, index = 0, always_accept = False):
        scenarios = self.scenarios(0, index)
        v = self.fn(state, scenarios)
        current, correction = self.fn.proposal(state, index) # need to pass level 0 scenario [=index] in case proposal depends on scenario
        v_prime = self.fn(current, scenarios)
        vdiff = (v_prime - v)/self.Ts[0] # these are cached evaluations, no side effects
        accept = True if always_accept else self.metropolis_accept(vdiff - correction)
        (self.acceptances if accept else self.rejections)[0] += 1
        return((current, vdiff) if accept else (state, 0.0))

