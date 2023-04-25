from ..base import types
from ..base.random import RNG
from scipy.stats import nbinom
import numpy as np
import copy


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

        if self.list_of_coin_tosses != other.list_of_coin_tosses:
            return False

        return True


# CoinStackProposal inherits from DiscreteVariableProposal
class CoinStackProposal(types.DiscreteVariableProposal):
    def __init__(self, start: CoinStack, rng=RNG()):
        self.start = start
        self.probs = [0.4, 0.4, 0.2]  # add, remove, change
        self.probs1 = [0.8, 0.0, 0.2]  # case where only one coin
        self.cumulative_probs = np.cumsum(self.probs)  # add, remove, change
        self.cumulative_probs1 = np.cumsum(self.probs1)  # add, remove, change
        self.rng = rng

    def eval(self, x):
        probs = self.probs
        if len(x.list_of_coin_tosses) == 1:
            probs = self.probs1  # only one coin, can only have add or change

        if len(x.list_of_coin_tosses) > len(self.start.list_of_coin_tosses):
            return np.log(0.5) + np.log(probs[0])
        elif len(x.list_of_coin_tosses) < len(self.start.list_of_coin_tosses):
            return np.log(probs[1])
        elif x != self.start:
            return np.log(probs[2])
        else:
            return -np.inf

    def sample(self):
        new_list_of_tosses = copy.deepcopy(self.start.list_of_coin_tosses)
        r = self.rng.random()
        cumulative_probs = self.cumulative_probs
        if len(self.start.list_of_coin_tosses) == 1:
            # Only one coin, can only append or change
            cumulative_probs = self.cumulative_probs1

        if r < cumulative_probs[0]:
            # append
            new_list_of_tosses.append(self.rng.randomInt(0, 1))
        elif r < cumulative_probs[1]:
            # remove
            del new_list_of_tosses[-1]
        else:
            # change
            index = self.rng.randomInt(0, len(new_list_of_tosses)-1)
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
            num_coins = self.rng.randomInt(1, self.a)
        else:
            num_coins = self.rng.randomInt(self.a+1, self.b)

        list_of_tosses = [self.rng.randomInt(0, 1) for i in range(num_coins)]
        return CoinStack(list_of_tosses)

    def eval(self, x: CoinStack):
        logprob = 0
        N = len(x.list_of_coin_tosses)
        if N <= self.a:
            logprob += np.log(self.p)
            logprob += -np.log(self.a-1+1)
        else:
            logprob += np.log(1 - self.p)
            logprob += -np.log(self.b-self.a+1)

        logprob = logprob - N * np.log(2)

        return logprob


# Uncertain counting of the number of heads
class CoinStackTarget(types.DiscreteVariableTarget):
    def __init__(self, mu, sigma):
        # NB as an over-dispersed Poisson
        self.p = mu/(sigma*sigma)
        self.n = mu*mu/(sigma*sigma - mu)

    def eval(self, x: CoinStack):
        # Evaluate logposterior at point x, P(x|D) \propto P(D|x)P(x)
        num_heads = sum(x.list_of_coin_tosses)
        target = nbinom.logpmf(num_heads, self.n, self.p)
        return target
