import numpy as np
from discretesampling.base.types import DiscreteVariable
from discretesampling.base.output import BaseOutput


class MCMCOutput(BaseOutput):
    def __init__(
        self,
        samples: list[DiscreteVariable],
        acceptance_probabilities,
        include_warmup,
        N=None, N_warmup=None
    ):
        super().__init__(samples)
        self.acceptance_probabilities = acceptance_probabilities
        self.include_warmup = include_warmup
        if N is None:
            N = len(samples)
        self.N = N
        self.N_warmup = N_warmup
        self.acceptance_rate = np.mean(acceptance_probabilities)
