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
import pandas as pd

# df = pd.read_csv(r"C:\Users\efthi\OneDrive\Desktop\PhD\datasets_smc_mcmc_CART\abalone.csv")

# y = df.Target
# X = df.drop(['Target'], axis=1)
# X = X.to_numpy()
# y = y.to_numpy()

data = datasets.load_wine()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)

a = 0.01
b = 5
target = dt.TreeTarget(a, b)
initialProposal = dt.TreeInitialProposal(X_train, y_train)

dtMCMC = DiscreteVariableMCMC(dt.Tree, target, initialProposal)
try:
    treeSamples = dtMCMC.sample(500)

    mcmcLabels = dt.stats(treeSamples, X_test).predict(X_test, use_majority=True)
    mcmcAccuracy = [dt.accuracy(y_test, mcmcLabels)]
    print("MCMC mean accuracy: ", (mcmcAccuracy))
except ZeroDivisionError:
    print("MCMC sampling failed due to division by zero")


dtSMC = DiscreteVariableSMC(dt.Tree, target, initialProposal)
try:
    treeSMCSamples = dtSMC.sample(10, 1000)

    smcLabels = dt.stats(treeSMCSamples, X_test).predict(X_test, use_majority=True)
    smcAccuracy = [dt.accuracy(y_test, smcLabels)]
    #majority_voting_acc = [dt.accuracy(y_test, majority_voting_labels)]
    print("SMC mean accuracy: ", np.mean(smcAccuracy))
    #print("SMC majority accuracy: ", (majority_voting_acc))
    
except ZeroDivisionError:
    print("SMC sampling failed due to division by zero")
