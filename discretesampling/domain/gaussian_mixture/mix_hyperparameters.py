from scipy.stats import beta, gamma, norm
import numpy as np
from numpy import random as rnd

class UnivariateMixHyperparameters:

    def __init__(self, g, alpha, la):
        self.alpha = alpha
        self.la = la
        self.g = g
        self.data = data


