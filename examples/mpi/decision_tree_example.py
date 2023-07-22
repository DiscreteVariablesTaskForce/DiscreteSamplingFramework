# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:28:12 2021
@author: efthi
"""
from discretesampling.base.util import gather_all
from mpi4py import MPI
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from discretesampling.base.algorithms import DiscreteVariableSMC
from discretesampling.domain import decision_tree as dt
import sys
sys.path.append('../')  # Looks like mpiexec won't find discretesampling package without appending '../'


data = datasets.load_wine()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)

a = 0.01
b = 5
target = dt.TreeTarget(a, b)
initialProposal = dt.TreeInitialProposal(X_train, y_train)

N = 1 << 10
T = 10
num_MC_runs = 1

dtSMC = DiscreteVariableSMC(dt.Tree, target, initialProposal)
try:
    runtimes = []
    accuracies = []
    for seed in range(num_MC_runs):
        MPI.COMM_WORLD.Barrier()
        start = MPI.Wtime()
        # seed = np.random.randint(0, 32766)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("seed = ", seed)
        treeSMCSamples = dtSMC.sample(T, N, seed)
        MPI.COMM_WORLD.Barrier()
        end = MPI.Wtime()
        if MPI.COMM_WORLD.Get_size() > 1:
            treeSMCSamples = gather_all(treeSMCSamples)

        smcLabels = [dt.stats(x, X_test).predict(X_test) for x in treeSMCSamples]
        smcAccuracy = [dt.accuracy(y_test, x) for x in smcLabels]

        if MPI.COMM_WORLD.Get_rank() == 0:
            accuracies.append(np.mean(smcAccuracy))  # replace mean with majority voting
            runtimes.append(end-start)
            print("SMC mean accuracy: ", accuracies[-1])
            print("SMC run-time: ", runtimes[-1])
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("SMC mean of mean accuracies: ", np.mean(accuracies))
        print("SMC median runtime: ", np.median(runtimes))
except ZeroDivisionError:
    print("SMC sampling failed due to division by zero")
