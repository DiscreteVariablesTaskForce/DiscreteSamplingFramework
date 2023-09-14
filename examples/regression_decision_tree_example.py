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

# data = datasets.load_wine()

# X = data.data
# y = data.target



import pandas as pd
df = pd.read_csv(r"C:\Users\efthi\OneDrive\Desktop\PhD\regression_datasets\Walmart.csv")
#df = df.sample(n=1000)
df=df.drop(["Date"], axis = 1)
df = df.dropna()
# df.day=df.day.astype('category').cat.codes
# df.month=df.month.astype('category').cat.codes


y = df.Unemployment
X = df.drop(['Unemployment'], axis=1)
X = X.to_numpy()
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)

a = 0.01
b = 5
target = dt.RegressionTreeTarget(a, b)
initialProposal = dt.TreeInitialProposal(X_train, y_train)

dtMCMC = DiscreteVariableMCMC(dt.Tree, target, initialProposal)
try:
    treeSamples = dtMCMC.sample(100)

    mcmcLabels = dt.RegressionStats(treeSamples, X_test).predict(X_test, use_majority=True)
    mcmcAccuracy = [dt.accuracy_mse(y_test, mcmcLabels) ]

    print("MCMC mean MSE: ", (mcmcAccuracy))
except ZeroDivisionError:
    print("MCMC sampling failed due to division by zero")


dtSMC = DiscreteVariableSMC(dt.Tree, target, initialProposal)
try:
    treeSMCSamples = dtSMC.sample(10, 10)

    smcLabels = dt.RegressionStats(treeSMCSamples, X_test).predict(X_test, use_majority=True)
    smcAccuracy = [dt.accuracy_mse(y_test, smcLabels)]

    print("SMC mean MSE: ", (smcAccuracy))
except ZeroDivisionError:
    print("SMC sampling failed due to division by zero")
