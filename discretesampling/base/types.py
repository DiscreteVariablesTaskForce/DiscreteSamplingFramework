from abc import ABC, abstractmethod
from typing import Union
import math
from pickle import loads, dumps
import numpy as np
from discretesampling.base.random import RNG
from discretesampling.base.kernel import DiscreteVariableOptimalLKernel


class DiscreteVariable(ABC):
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def getProposalType(self):
        pass

    @classmethod
    @abstractmethod
    def getTargetType(self):
        pass

    @classmethod
    def getLKernelType(self):
        # Forward proposal
        return self.getProposalType()

    @classmethod
    def getOptimalLKernelType(self):
        return DiscreteVariableOptimalLKernel

    @classmethod
    def encode(self, x):
        encoded = np.array(bytearray(dumps(x)))
        return encoded

    @classmethod
    def decode(self, x, particle):
        pickle_stopcode = 0x2e
        end_of_pickle_data = np.argwhere(x == pickle_stopcode)[-1][0] + 1
        encoded = np.array(x[0:end_of_pickle_data], dtype=np.uint8)
        decoded = loads(bytes(encoded))
        return decoded


class DiscreteVariableProposal(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def norm(self, x):
        pass

    # Should return true if proposal is possible between x and y
    # (and possibly at other times)
    # where x and y are norm values from the above function
    @abstractmethod
    def heuristic(self, x, y):
        pass

    @abstractmethod
    def sample(
        self,
        x: DiscreteVariable,
        rng: RNG = RNG(),
        target: Union[None, 'DiscreteVariableTarget'] = None
    ) -> 'DiscreteVariable':
        pass

    @abstractmethod
    def eval(
        self,
        x: DiscreteVariable,
        x_prime: DiscreteVariable,
        target: Union[None, 'DiscreteVariableTarget'] = None
    ) -> float:
        pass


# Exact same as proposal above
class DiscreteVariableInitialProposal(ABC):
    def __init__(self, values, probs):
        # Check dims and probs are valid
        assert len(values) == len(probs), "Invalid PMF specified, x and p" +\
            " of different lengths"
        probs = np.array(probs)
        tolerance = np.sqrt(np.finfo(np.float64).eps)
        assert abs(1 - sum(probs)) < tolerance, "Invalid PMF specified," +\
            " sum of probabilities !~= 1.0"
        assert all(probs > 0), "Invalid PMF specified, all probabilities" +\
            " must be > 0"
        self.x = values
        self.pmf = probs
        self.cmf = np.cumsum(probs)

    def sample(self, rng: RNG = RNG(), target: Union[None, 'DiscreteVariableTarget'] = None) -> DiscreteVariable:
        q = rng.random()  # random unif(0,1)
        return self.x[np.argmax(self.cmf >= q)]

    def eval(self, y: DiscreteVariable, target: Union[None, 'DiscreteVariableTarget'] = None) -> float:
        try:
            i = self.x.index(y)
            logp = math.log(self.pmf[i])
        except ValueError:
            print("Warning: value " + str(y) + " not in pmf")
            logp = -math.inf
        return logp


class DiscreteVariableTarget(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def eval(self, x: DiscreteVariable) -> float:
        pass

    def evaluatePrior(self, x: DiscreteVariable) -> float:
        pass
