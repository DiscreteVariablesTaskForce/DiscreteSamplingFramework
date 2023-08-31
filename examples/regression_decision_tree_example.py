# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:28:12 2021

@author: efthi
"""

from discretesampling.domain import decision_tree as dt
from discretesampling.base.algorithms import DiscreteVariableMCMC, DiscreteVariableSMC

from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np

data = datasets.load_wine()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)

a = 0.01
b = 5
target = dt.RegressionTreeTarget(a, b)
initialProposal = dt.TreeInitialProposal(X_train, y_train)

dtMCMC = DiscreteVariableMCMC(dt.Tree, target, initialProposal)
try:
    treeSamples = dtMCMC.sample(500)

    mcmcLabels = [dt.RegressionStats(x, X_test).regression_predict(X_test) for x in treeSamples]
    mcmcAccuracy = [dt.accuracy_mse(y_test, x) for x in mcmcLabels]

    print("MCMC mean MSE: ", np.mean(mcmcAccuracy[250:499]))
except ZeroDivisionError:
    print("MCMC sampling failed due to division by zero")


dtSMC = DiscreteVariableSMC(dt.Tree, target, initialProposal)
try:
    treeSMCSamples = dtSMC.sample(1024, 10)

    smcLabels = [dt.RegressionStats(x, X_test).regression_predict(X_test) for x in treeSMCSamples]
    smcAccuracy = [dt.accuracy_mse(y_test, x) for x in smcLabels]

    print("SMC mean MSE: ", np.mean(smcAccuracy))
except ZeroDivisionError:
    print("SMC sampling failed due to division by zero")
