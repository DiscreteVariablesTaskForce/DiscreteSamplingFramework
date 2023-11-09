from abc import ABC
from discretesampling.base.random import RNG


class ContinuousProposal(ABC):
    def __init__(self):
        pass

    def sample(self, x, params, y, rng: RNG = RNG()):
        pass

    def eval(self, x, params, y, p_params):
        pass
