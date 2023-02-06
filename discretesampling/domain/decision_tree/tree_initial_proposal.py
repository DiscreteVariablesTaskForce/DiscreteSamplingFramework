from discretesampling.base.random import RNG
from ...base.types import DiscreteVariableInitialProposal
from .tree import Tree
import math


class TreeInitialProposal(DiscreteVariableInitialProposal):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        #self.rng = rng

    def sample(self, rng):
        leafs = [1, 2]

        feature = rng.randomInt(0, len(self.X_train[0])-1)
        threshold = rng.randomInt(0, len(self.X_train)-1)
        tree = [[0, 1, 2, feature, self.X_train[threshold, feature]]]
        return Tree(self.X_train, self.y_train, tree, leafs)

    def eval(self, x):
        num_features = len(self.X_train[0])
        num_thresholds = len(self.X_train)
        return -math.log(num_features) - math.log(num_thresholds)
