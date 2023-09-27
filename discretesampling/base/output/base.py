from abc import ABC
import numpy as np


class BaseOutput(ABC):
    def __init__(self, samples):
        self.samples = samples
        self.log_weights = np.repeat([-len(samples)], len(samples))
