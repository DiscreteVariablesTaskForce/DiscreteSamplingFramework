from abc import abc
import numpy as np


class BaseOutput(abc):
    def __init__(self, samples):
        self.samples = samples
        self.log_weights = np.repeat([-len(samples)], len(samples))
