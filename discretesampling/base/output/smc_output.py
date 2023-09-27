import numpy as np
from discretesampling.base.types import DiscreteVariable
from discretesampling.base.output import BaseOutput
from discretesampling.base.algorithms.smc_components import ess
from discretesampling.base.executor import Executor


class SMCOutput(BaseOutput):
    def __init__(
        self,
        samples: list[DiscreteVariable],
        log_weights: np.ndarray,
        exec: Executor = Executor()
    ):
        super().__init__(self, samples)
        self.log_weights = log_weights
        self.ess = ess(log_weights, exec=exec)
