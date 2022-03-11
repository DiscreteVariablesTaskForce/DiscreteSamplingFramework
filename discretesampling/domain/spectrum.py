from ..base import types
import math


# SpectrumDimension inherits from DiscreteVariable
class SpectrumDimension(types.DiscreteVariable):
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
class SpectrumDimensionProposal(types.DiscreteVariableProposal):
    def __init__(self, startingDimension: SpectrumDimension):
        startingValue = startingDimension.value

        if startingValue > 0:
            firstValue = startingValue - 1
        else:
            firstValue = 0

        dims = [SpectrumDimension(x) for x in range(firstValue,
                                                    startingValue+2)]
        numDims = len(dims)
        probs = [1/numDims] * numDims

        super().__init__(dims, probs)

    @classmethod
    def norm(self, x):
        return x.value

    @classmethod
    def heuristic(self, x, y):
        # Proposal can move at most one value up or down
        return abs(y-x) < 2


class SpectrumDimensionTarget(types.DiscreteVariableTarget):
    def __init__(self, data):
        self.data = data

    def eval(self, x):
        # Evaluate logposterior at point x, P(x|D) \propto P(D|x)P(x)
        return -math.inf
