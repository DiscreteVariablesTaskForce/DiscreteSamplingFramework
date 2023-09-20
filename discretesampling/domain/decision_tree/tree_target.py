from typing import TYPE_CHECKING
import math
import numpy as np
from ...base import types
from .metrics import calculate_leaf_occurences

if TYPE_CHECKING:
    from discretesampling.domain.decision_tree import Tree


class TreeTarget(types.DiscreteVariableTarget):
    def __init__(self, a, b=None):
        self.a = a
        self.b = b

    def eval(self, x):
        # call test tree to calculate Π(Y_i|T,theta,x_i)
        target1, leafs_possibilities_for_prediction = calculate_leaf_occurences(x)
        # call test tree to calculate  (theta|T)
        target2 = self.features_and_threshold_probabilities(x)
        # p(T)
        target3 = self.evaluatePrior(x)
        return (target1+target2+target3)

    # (theta|T)
    def features_and_threshold_probabilities(self, x: 'Tree'):
        """Calculate the probability of selecting the given set of features and
        thresholds in the Tree `x` from the training data.

        Parameters
        ----------
        x : Tree
            Tree to evaluate

        Returns
        -------
        float
            log-probability of selecting the given features and thresholds

        Notes
        -----
        The likelihood of choosing the specific feature and threshold must
        be computed. We need to find out the probabilty of selecting the
        specific feature multiplied by 1/the margins. It should be
        (1/number of features) * (1/(upper bound-lower bound))
        """
        # this need to change
        logprobabilities = []

        for node in x.tree:
            logprobabilities.append(-math.log(len(x.X_train[0]))
                                    - math.log(max(x.X_train[:, node[3]]) - min(x.X_train[:, node[3]]))
                                    )

        logprobability = np.sum(logprobabilities)
        return (logprobability)

    def evaluatePrior(self, x):
        if self.b is None:
            # Use Possion prior
            return self.evaluatePoissonPrior(x)
        else:
            return self.evaluateChipmanPrior(x)

    def evaluatePoissonPrior(self, x):
        # From 'A Bayesian CART algorithm' -  Densison et al. 1998
        lam = self.a
        k = len(x.leafs)
        logprior = math.log(math.pow(lam, k) / ((math.exp(lam)-1) * math.factorial(k)))

        return logprior

    def evaluateChipmanPrior(self, x):
        # From 'Bayesian CART model search' - Chipman et al. 1998
        def p_node(a, b, d):
            return math.log(a / math.pow(1 + d, b))

        def p_leaf(a, b, d):
            return math.log(1 - math.exp(p_node(a, b, d)))

        logprior = 0
        for node in x.tree:
            logprior += p_node(self.a, self.b, node[5])

        for leaf in x.leafs:
            d = x.depth_of_leaf(leaf)
            logprior += p_leaf(self.a, self.b, d)
        return logprior
