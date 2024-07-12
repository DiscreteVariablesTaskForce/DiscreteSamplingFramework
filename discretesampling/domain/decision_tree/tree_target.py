import math
import numpy as np
# from discretesampling.base.types import DiscreteVariableTarget
from discretesampling.domain.decision_tree.metrics import calculate_leaf_occurences

# number of possible binary trees with n decision nodes:
def catalan(n):
    return 1 if (n<=1) else ((2 * (n + n - 1) * catalan(n-1))//(n+1))
# NOTE 2 trees will have the same IDs and leaves if they are the same w.r.t. this count


# class TreeTarget(DiscreteVariableTarget):
class TreeTarget:
    def __init__(self, a, b = None): # a only used for Poisson prior, b only used for Chipman
        self.a = a
        self.b = b
    #
    def eval(self, x, X_train, y_train):
        print ("WARNING: DO NOT USE FOR HINTS DUE TO FEATURE SHAPE")
        # call test tree to calculate Π(Y_i|T,theta,x_i)
        target1, _ = calculate_leaf_occurences(x, X_train, y_train)
        # call test tree to calculate  (theta|T)
        target2 = self.features_and_threshold_probabilities(x, X_train)
        # p(T)
        target3 = self.evaluatePrior(x)
        return (target1+target2+target3)

    # (theta|T)
    # MJAS use continuous if you are going to draw thresholds uniformly
    # otherwise they are drawn from feature values in the (WHOLE) training set
    # MJAS the prior was also incorrect unless we have a fully balanced tree... we need to specify which leaf was expanded to grow each node
    # hence  the catalan factor
    def features_and_threshold_probabilities(self, x, features, continuous_thresholds = False, verbose = False):
        # this need to change
        fshape = features.shape
        logprobabilities = []
        '''
            original comment:
            the likelihood of choosing the specific feature and threshold must
            be computed. We need to find out the probabilty of selecting the
            specific feature multiplied by 1/the margins. it should be
            (1/number of features) * (1/(upper bound-lower bound))
        '''
        
        for node in x.tree: # the root node can also change (but not be removed)
            lp = -math.log(fshape[1]) # choice of feature at this node
            if continuous_thresholds: # choice of threshold at this node
                relevant = features[:, node[3]]
                lp -= math.log(max(relevant) - min(relevant))
            else:
                lp -= math.log(fshape[0])
                if verbose:
                    print(ni + 2, fshape[1], fshape[0], np.round(np.exp(-lp)))
            logprobabilities.append(lp)
        #
        logprobability = np.sum(logprobabilities) - math.log(catalan(len(x.tree))) # second term accounts for all the possible tree shapes (different once thresholds are placed in them) 

        if verbose:
            print(np.round(np.exp(-logprobability)))
        return (logprobability)

    def evaluatePrior(self, x):
        if self.b is None:
            # Use Possion prior
            #return self.evaluateGeometricPrior(x) # could be better than Poisson but choosing a is difficult
            return self.evaluatePoissonPrior(x)
        else:
            return self.evaluateChipmanPrior(x)
    
    # MJAS this PoissonPrior is incomplete as a prior if you don't somewhere assign probability to the different tree branching options
    # e.g. in the features_and_thresholds part of the prior
    def evaluatePoissonPrior(self, x):
        # From 'A Bayesian CART algorithm' -  Densison et al. 1998
        lam = self.a
        k = len(x.leafs)
        logprior = math.log(math.pow(lam, k) / ((math.exp(lam)-1) * math.factorial(k)))
        return logprior

    def evaluateGeometricPrior(self, x):
        #q = 1.0/self.a
        #p = 1.0 - q
        k = len(x.leafs)
        logprior = (k-1) * (-math.log(self.a)) + math.log(self.a - 1.0) - math.log(self.a)
        return logprior

    
    def evaluateChipmanPrior(self, x):
        # From 'Bayesian CART model search' - Chipman et al. 1998
        # MJAS - this prior may be better but need to check what node[5] and make sure it properly assigns prob to all tree shapes
        #      - if using this then probably you need to remove the tree shape prior in features_and_thresholds
        raise Exception("Needs testing - see comments")
        def p_node(a, b, d):
            return math.log(a / math.pow(1 + d, b))

        def p_leaf(a, b, d):
            return math.log(1 - math.exp(p_node(a, b, d)))
        
        logprior = 0
        for node in x.tree:
            logprior += p_node(self.a, self.b, node[5]) # this is the only place node depth is used

        for leaf in x.leafs:
            d = x.depth_of_leaf(leaf)
            logprior += p_leaf(self.a, self.b, d)
        return logprior
