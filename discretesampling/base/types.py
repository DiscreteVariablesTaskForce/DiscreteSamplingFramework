import math
from pickle import loads, dumps
import numpy as np
from discretesampling.base.random import RNG
from discretesampling.base.kernel import DiscreteVariableOptimalLKernel


class DiscreteVariable:
    def __init__(self):
        pass

    @classmethod
    def getProposalType(self):
        return DiscreteVariableProposal

    @classmethod
    def getTargetType(self):
        return DiscreteVariableTarget

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
        encoded_size = encoded.size
        encoded_size_bytes = encoded_size.to_bytes((encoded_size.bit_length() + 7) // 8, "big")
        encoded = np.hstack(
            (np.array(bytearray(encoded_size_bytes)),  # size of encoded object in bytes
             [np.uint8(0)],  # zero byte to mark end of size
             encoded)  # encoded object
        )
        return encoded

    @classmethod
    def decode(self, x, particle):
        zero_index = np.argwhere(x == 0)[0][0]
        size_bytes = x[0:zero_index]
        encoded_size = int.from_bytes(bytes(size_bytes), "big")
        encoded = x[(zero_index+1):(zero_index+1+encoded_size)]
        decoded = loads(bytes(encoded))
        return decoded


class DiscreteVariableProposal:
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

    @classmethod
    def norm(self, x):
        return 1

    @classmethod
    # Should return true if proposal is possible between x and y
    # (and possibly at other times)
    # where x and y are norm values from the above function
    def heuristic(self, x, y):
        return True

    def sample(self, rng=RNG(), target=None):
        q = rng.random()  # random unif(0,1)
        return self.x[np.argmax(self.cmf >= q)]

    def eval(self, y, target=None):
        try:
            i = self.x.index(y)
            logp = math.log(self.pmf[i])
        except ValueError:
            print("Warning: value " + str(y) + " not in pmf")
            logp = -math.inf
        return logp


# Exact same as proposal above
class DiscreteVariableInitialProposal(DiscreteVariableProposal):
    def sample(self, rng=RNG(), target=None):
        return super().sample(rng)


class DiscreteVariableTarget:
    def __init__(self):
        pass

    def eval(self, x):
        logprob = -math.inf
        logPrior = self.evaluatePrior(x)
        logprob += logPrior
        return logprob

    def evaluatePrior(self, x):
        logprob = -math.inf
        return logprob
