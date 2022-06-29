from math import log, pi, pow
from numpy import sum, prod
from .numbers import bell
from ...base import types


class AdditiveStructureTarget(types.DiscreteVariableTarget):
    def __init__(self, data):
        # self.data = data
        self.xtrain = data[0]
        self.ytrain = data[1]

    def eval(self, x):
        # Calculate logposterior at "point" x, an instance of AdditiveStructure
        # presumably some function of x.discrete_set and some data which
        # could be defined in constructor as self.data

        D = self.xtrain.shape[1]
        y = [self.evaluate(x.discrete_set, self.xtrain.iloc[i]) for i in range(self.xtrain.shape[0])]
        cov = 0.25  # np.eye(D) * np.power(sigma,2)

        # calculate Π(Y_i|M,theta,x_i) # make data from gaussian with structure (?)
        target1 = [self.log_likelihood(self.ytrain.iloc[i], y[i], cov) for i in range(len(y))]
        targ = sum(target1)

        # p(M)
        target2 = self.evaluatePrior(x)
        target2 = log(target2)
        return targ + target2

    def evaluatePrior(self, x):
        n = len(x.discrete_set)
        return 1 / bell(n)

    def log_likelihood(self, y, mean, var):
        return -log(var) - (0.5 * log(2 * pi)) - (0.5 * pow((y - mean) / var, 2))

    def evaluate(self, structure, data):
        y = [0] * len(structure)
        data.reset_index(inplace=True, drop=True)
        for subset in structure:
            i = 0
            for j in subset:
                y[i] = y[i] + data[j-1]
        return prod(y)
