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

# Poisson prior on the number of LEAVES:
#   log P(k) = k*log(a) - log(e^a - 1) - log(k!)
# See incremental_decision_tree_example.py, which is set up to match this.
a = 15
b = None
target = dt.TreeTarget(a, b)
initialProposal = dt.TreeInitialProposal(X_train, y_train)

dtMCMC = DiscreteVariableMCMC(dt.Tree, target, initialProposal)
try:
    treeSamples = dtMCMC.sample(20_000)

    mcmcLabels = dt.stats(treeSamples[19_000:], X_test).predict(X_test, use_majority=True)
    # dt.accuracy already returns a percentage
    mcmcAccuracy = dt.accuracy(y_test, mcmcLabels)
    print("MCMC acceptance rate:  %.3f (last 1000: %.3f)"
          % (dtMCMC.acceptance_rate, dtMCMC.tail_acceptance_rate))
    print("MCMC mean leaves:      %.1f"
          % np.mean([len(t.leafs) for t in treeSamples[19_000:]]))
    print("MCMC mean accuracy:    %.2f%%" % mcmcAccuracy)
except ZeroDivisionError:
    print("MCMC sampling failed due to division by zero")


dtSMC = DiscreteVariableSMC(dt.Tree, target, initialProposal)
try:
    treeSMCSamples = dtSMC.sample(10, 1000)

    smcLabels = dt.stats(treeSMCSamples, X_test).predict(X_test, use_majority=True)
    smcAccuracy = [dt.accuracy(y_test, smcLabels)]
    print("SMC mean accuracy: ", np.mean(smcAccuracy))

except ZeroDivisionError:
    print("SMC sampling failed due to division by zero")
