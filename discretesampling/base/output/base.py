from abc import ABC
import numpy as np


class BaseOutput(ABC):
    def __init__(self, samples):
        self.samples = samples
        self.log_weights = np.repeat([-len(samples)], len(samples))

    def __eq__(self, other):
        samples_eq = np.array_equal(self.samples, other.samples)
        weights_eq = np.array_equal(self.log_weights, other.log_weights)
        return samples_eq and weights_eq
