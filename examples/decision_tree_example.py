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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style("whitegrid")

# data = datasets.load_wine()

# X = data.data
# y = data.target

X, y = datasets.fetch_covtype(return_X_y=True)
y = pd.Categorical(y).codes

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=0)

# Poisson prior on the number of LEAVES:
#   log P(k) = k*log(a) - log(e^a - 1) - log(k!)
# See incremental_decision_tree_example.py, which is set up to match this.
a = 15
b = None
target = dt.TreeTarget(a, b)
initialProposal = dt.TreeInitialProposal(X_train, y_train)

# dtMCMC = DiscreteVariableMCMC(dt.Tree, target, initialProposal)
# try:
#     treeSamples = dtMCMC.sample(20_000, keep_samples=True)

#     mcmcLabels = dt.stats(treeSamples[19_000:], X_test).predict(X_test, use_majority=True)
#     # dt.accuracy already returns a percentage
#     mcmcAccuracy = dt.accuracy(y_test, mcmcLabels)
#     print("MCMC acceptance rate:  %.3f (last 1000: %.3f)"
#           % (dtMCMC.acceptance_rate, dtMCMC.tail_acceptance_rate))
#     print("MCMC mean leaves:      %.1f"
#           % np.mean([len(t.leafs) for t in treeSamples[19_000:]]))
#     print("MCMC mean accuracy:    %.2f%%" % mcmcAccuracy)
# except ZeroDivisionError:
#     print("MCMC sampling failed due to division by zero")


dtSMC = DiscreteVariableSMC(dt.Tree, target, initialProposal)
try:
    treeSMCSamples = dtSMC.sample(20, 10)

    smcLabels = dt.stats(treeSMCSamples, X_test).predict(X_test, use_majority=True)
    smcAccuracy = [dt.accuracy(y_test, smcLabels)]
    print("SMC mean accuracy: ", np.mean(smcAccuracy))
    plt.figure()
    plt.plot(dtSMC.ess_history)
    plt.title("SMC Effective Sample Size")
    plt.xlabel("Iteration")
    plt.ylabel("Effective Sample Size")
    plt.show()

except ZeroDivisionError:
    print("SMC sampling failed due to division by zero")
