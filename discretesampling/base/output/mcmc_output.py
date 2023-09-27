import numpy as np
from discretesampling.base.types import DiscreteVariable
from discretesampling.base.output import BaseOutput


class MCMCOutput(BaseOutput):
    def __init__(
        self,
        samples: list[DiscreteVariable],
        acceptance_probabilities=None,
        include_warmup=None,
        N=None, N_warmup=None
    ):
        super().__init__(samples)
        self.acceptance_probabilities = acceptance_probabilities
        self.include_warmup = include_warmup
        if N is None:
            N = len(samples)
        self.N = N
        self.N_warmup = N_warmup
        acceptance_rate = None
        if acceptance_probabilities is not None:
            acceptance_rate = np.mean(acceptance_probabilities)
        self.acceptance_rate = acceptance_rate
