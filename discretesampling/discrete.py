import numpy as np
import random
import math
from scipy.special import logsumexp


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

    def sample(self):
        q = random.random()  # random unif(0,1)
        return self.x[np.argmax(self.cmf >= q)]

    def eval(self, y):
        try:
            i = self.x.index(y)
            logp = math.log(self.pmf[i])
        except ValueError:
            print("Warning: value " + str(y) + " not in pmf")
            logp = -math.inf
        return logp


# Exact same as proposal above
class DiscreteVariableInitialProposal:
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

    def sample(self):
        q = random.random()  # random unif(0,1)
        return self.x[np.argmax(self.cmf >= q)]

    def eval(self, y):
        try:
            i = self.x.index(y)
            logp = math.log(self.pmf[i])
        except ValueError:
            print("Warning: value " + str(y) + " not in pmf")
            logp = -math.inf
        return logp


class DiscreteVariableTarget:
    def __init__(self):
        pass

    def eval(self, x):
        logprob = -math.inf
        return logprob


class DiscreteVariableOptimalLKernel:
    def __init__(self):
        pass

    def eval(self, current_particle, previous_particles, p):
        proposalType = type(current_particle).getProposalType()
        logprob = -math.inf

        forward_probabilities = np.zeros(len(previous_particles))
        eta = np.zeros(len(previous_particles))
        for i in range(len(previous_particles)):
            forward_proposal = proposalType(previous_particles[i])
            forward_probabilities[i] = forward_proposal.eval(current_particle)

            eta[i] = previous_particles.count(previous_particles[i]) / len(previous_particles)

        eta_numerator = eta[p]
        forward_probability_numerator = forward_probabilities[p]

        numerator = forward_probability_numerator + math.log(eta_numerator)
        denominator_p = np.array(
            [forward_probabilities[i] + math.log(eta[i]) for i in range(len(forward_probabilities))])
        denominator = logsumexp(np.setdiff1d(denominator_p, -math.inf))

        logprob = numerator - denominator

        return logprob

