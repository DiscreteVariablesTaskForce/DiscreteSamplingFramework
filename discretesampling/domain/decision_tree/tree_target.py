import math
import numpy as np
from discretesampling.base.types import DiscreteVariableTarget
from discretesampling.domain.decision_tree.metrics import calculate_leaf_occurences


def catalan(n):
    return 1 if (n<=1) else ((2 * (n + n - 1) * catalan(n-1))//(n+1))

class TreeTarget(DiscreteVariableTarget):
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
    def features_and_threshold_probabilities(self, x, continuous_thresholds = False, verbose = False):
        # this need to change
        logprobabilities = []
        fshape = x.X_train.shape

        '''
            the likelihood of choosing the specific feature and threshold must
            be computed. We need to find out the probabilty of selecting the
            specific feature multiplied by 1/the margins. it should be
            (1/number of features) * (1/(upper bound-lower bound))
        '''
        # for node in x.tree:
        #     logprobabilities.append(-math.log(len(x.X_train[0]))
        #                             - math.log(max(x.X_train[:, node[3]]) - min(x.X_train[:, node[3]]))
        #                             )

        # logprobability = np.sum(logprobabilities)
        
        for node in x.tree: # the root node can also change (but not be removed)
            lp = -math.log(fshape[1]) # choice of feature at this node
            if continuous_thresholds: # choice of threshold at this node
                relevant = x.X_train[:, node[3]]
                lp -= math.log(max(relevant) - min(relevant))
            else:
                lp -= math.log(fshape[0])
                # if verbose:
                #     print(ni + 2, fshape[1], fshape[0], np.round(np.exp(-lp)))
            logprobabilities.append(lp)
        #
        logprobability = np.sum(logprobabilities) - math.log(catalan(len(x.tree))) # second term accounts for all the possible tree shapes (different once thresholds are placed in them) 

        # if verbose:
        #     print(np.round(np.exp(-logprobability)))
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
