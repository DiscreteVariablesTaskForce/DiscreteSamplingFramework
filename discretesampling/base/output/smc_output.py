import numpy as np
from typing import Union
from discretesampling.base.types import DiscreteVariable
from discretesampling.base.output import BaseOutput
from discretesampling.base.algorithms.smc_components import ess


class SMCOutput(BaseOutput):
    def __init__(
        self,
        samples: list[DiscreteVariable],
        log_weights: Union[np.ndarray, None] = None
    ):
        super().__init__(samples)
        if log_weights is not None:
            self.log_weights = log_weights
        self.ess = ess(self.log_weights)
