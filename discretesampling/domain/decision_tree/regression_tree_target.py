# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 09:24:14 2023

@author: efthi
"""

import math
import numpy as np
from discretesampling.domain.decision_tree.tree_target import TreeTarget
from discretesampling.domain.decision_tree.regression_metrics import regression_likelihood

class RegressionTreeTarget(TreeTarget):
    def eval(self, x):
        # call test tree to calculate Π(Y_i|T,theta,x_i)
        target1, leafs_possibilities_for_prediction = regression_likelihood(x)
        # call test tree to calculate  (theta|T)
        target2 = self.features_and_threshold_probabilities(x)
        # p(T)
        target3 = self.evaluatePrior(x)
        return (target1+target2+target3)
        
        