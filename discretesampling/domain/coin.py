from ..base import types
from ..base.random import RNG
from scipy.stats import poisson
import numpy as np


# CoinStack inherits from DiscreteVariable
class CoinStack(types.DiscreteVariable):
    def __init__(self, list_of_coin_tosses):
        super().__init__()
        self.list_of_coin_tosses = list_of_coin_tosses

    @classmethod
    def getProposalType(self):
        return CoinStackProposal

    @classmethod
    def getTargetType(self):
        return CoinStackTarget

    # Are equal if values are equal
    def __eq__(self, other):
        if not isinstance(other, CoinStack):
            return NotImplemented

        if self.value != other.value:
            return False

        return True


# CoinStackProposal inherits from DiscreteVariableProposal
class CoinStackProposal(types.DiscreteVariableProposal):
    def __init__(self, start: CoinStack, rng=RNG()):
        self.start = start
        self.probs = [0.4,0.4,0.2] # add, remove, change
        self.cumulative_probs = np.cumsum(self.probs) # add, remove, change
        self.rng = rng
    
    def eval(self, x):
        if len(x) > len(self.start.list_of_coin_tosses):
            return np.log(0.5) + np.log(self.probs[0])
        elif len(x) < len(self.start.list_of_coin_tosses):
            return np.log(self.probs[1])
        else:
            return np.log(self.probs[2])


    def sample(self):
        new_list_of_tosses = self.start.list_of_coin_tosses
        r = self.rng.random()
        if r < self.cumulative_probs[0]:
            #add
            new_list_of_tosses.append(self.rng.randomInt(0,1))
        elif r < self.cumulative_probs[1]:
            #remove
            del new_list_of_tosses[-1]
        else:
            #change
            index = self.rng.randomInt(0,len(new_list_of_tosses))
            new_list_of_tosses[index] = 1 - new_list_of_tosses[index]

        return CoinStack(new_list_of_tosses)

class CoinStackInitialProposal(types.DiscreteVariableProposal):
    def __init__(self, a, b, p,  rng=RNG()):
        self.a = a
        self.b = b
        self.p = p
        self.rng = rng

    def sample(self):
        r = self.rng.random()
        num_coins = 1
        if r < self.p:
            num_coins = self.rng.randomInf(1,self.a)
        else:    
            num_coins = self.rng.randomInf(self.a+1,self.b)
        
        list_of_tosses = [self.rng.randomInt(0,1) for i in range(num_coins)]
        return CoinStack(list_of_tosses)

    def eval(self, x: CoinStack):
        logprob = 0
        N = len(x.list_of_coin_tosses)
        if N <= self.a:
            logprob = logprob + np.log(self.p)
        else:
            logprob = logprob + np.log(1 - self.p)
        
        logprob = logprob + N * np.log(2)
        

#Uncertain counting of the number of heads
#Reported count of k
class CoinStackTarget(types.DiscreteVariableTarget):
    def __init__(self, k):        
        self.k = k

    def eval(self, x: CoinStack):
        # Evaluate logposterior at point x, P(x|D) \propto P(D|x)P(x)
        num_heads = sum(x.list_of_coin_tosses)

        target = poisson.logpmf(self.k, num_heads)
        return target
