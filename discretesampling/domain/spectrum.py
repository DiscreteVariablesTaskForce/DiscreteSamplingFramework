from scipy.stats import nbinom
import numpy as np
from discretesampling.base.types import (
    DiscreteVariable,
    DiscreteVariableTarget,
    DiscreteVariableInitialProposal,
    DiscreteVariableProposal
)
from discretesampling.base.random import RNG


# SpectrumDimension inherits from DiscreteVariable
class SpectrumDimension(DiscreteVariable):
    def __init__(self, value):
        super().__init__()
        self.value = value

    @classmethod
    def getProposalType(self):
        return SpectrumDimensionProposal

    @classmethod
    def getTargetType(self):
        return SpectrumDimensionTarget

    # Are equal if values are equal
    def __eq__(self, other):
        if not isinstance(other, SpectrumDimension):
            return NotImplemented

        if self.value != other.value:
            return False

        return True


# SpectrumDimensionProposal inherits from DiscreteVariableProposal
class SpectrumDimensionProposal(DiscreteVariableProposal):
    def __init__(self):
        pass

    def sample(self, startingDimension: SpectrumDimension, rng=RNG()):
        dims, pmf = self.generateDims(startingDimension)
        cmf = np.cumsum(pmf)
        q = rng.random()  # random unif(0,1)
        sampledDim = dims[np.argmax(cmf >= q)]
        return sampledDim

    def eval(self, startingDimension: SpectrumDimension, sampledDimension: SpectrumDimension):
        dims, pmf = self.generateDims(startingDimension)
        try:
            i = dims.index(sampledDimension)
            logp = np.log(pmf[i])
        except ValueError:
            print("Warning: value " + str(sampledDimension) + " not in pmf")
            logp = -np.inf
        return logp

    def generateDims(self, startingDim: SpectrumDimension):
        startingValue = startingDim.value
        values = []
        if startingValue > 1:
            values = [startingValue-1, startingValue+1]
        else:
            values = [startingValue+1]
        dims = [SpectrumDimension(x) for x in values]
        numDims = len(dims)
        probs = [1/numDims] * numDims
        return dims, probs

    @classmethod
    def norm(self, x):
        return x.value

    @classmethod
    def heuristic(self, x, y):
        # Proposal can move at most one value up or down
        return abs(y-x) == 1


class SpectrumDimensionInitialProposal(DiscreteVariableInitialProposal):
    def __init__(self, max):
        dims = [SpectrumDimension(x+1) for x in range(max)]
        numDims = len(dims)
        probs = [1/numDims] * numDims

        super().__init__(dims, probs)


class SpectrumDimensionTarget(DiscreteVariableTarget):
    def __init__(self, mu, sigma):
        # NB as an over-dispersed Poisson
        self.p = mu/(sigma*sigma)
        self.n = mu*mu/(sigma*sigma - mu)

    def eval(self, x):
        # Evaluate logposterior at point x, P(x|D) \propto P(D|x)P(x)
        target = self.evaluatePrior(x)
        return target

    def evaluatePrior(self, x):
        logprob = nbinom(self.n, self.p).logpmf(x.value)
        return logprob
