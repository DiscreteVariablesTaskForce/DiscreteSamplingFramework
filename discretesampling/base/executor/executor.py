import itertools
import numpy as np
from scipy.special import logsumexp


class Executor(object):
    def __init__(self):
        self.P = 1
        self.rank = 0

    def max(self, x):
        return np.max(x)

    def sum(self, x):
        return np.sum(x)

    def gather(self, x, all_x_shape):
        return x

    def bcast(self, x):
        pass

    def logsumexp(self, x):
        return logsumexp(x)

    def cumsum(self, x):
        return np.cumsum(x)

    def redistribute(self, particles, ncopies):
        particles = list(itertools.chain.from_iterable(
            [[particles[i]]*ncopies[i] for i in range(len(particles))]
        ))
        return particles
