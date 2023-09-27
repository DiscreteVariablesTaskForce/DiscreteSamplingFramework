import numpy as np
from discretesampling.base.types import DiscreteVariable
from discretesampling.base.output import BaseOutput
from discretesampling.base.algorithms.smc_components import ess


class SMCOutput(BaseOutput):
    def __init__(
        self,
        samples: list[DiscreteVariable],
        log_weights: np.ndarray
    ):
        super().__init__(samples)
        self.log_weights = log_weights
        self.ess = ess(log_weights)
