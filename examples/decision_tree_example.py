# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:28:12 2021
@author: efthi
"""
import sys
sys.path.append('../')  # Looks like mpiexec won't find discretesampling package without appending '../'
from discretesampling.domain import decision_tree as dt
from discretesampling.base.algorithms import DiscreteVariableSMC
from discretesampling.base.algorithms import DiscreteVariableMCMC

from sklearn import datasets
from sklearn.model_selection import train_test_split

import numpy as np
from mpi4py import MPI
from discretesampling.base.algorithms.smc_components.util import gather_all

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


dtMCMC = DiscreteVariableMCMC(dt.Tree, target, initialProposal)
try:
    runtimes = []
    accuracies = []
    for seed in range(num_MC_runs):
        MPI.COMM_WORLD.Barrier()
        start = MPI.Wtime()
        P = MPI.COMM_WORLD.Get_size()
        num_samples = int(N*T / P)
        rank = MPI.COMM_WORLD.Get_rank()
        treeSamples = dtMCMC.sample(num_samples, seed=seed+num_samples*rank)
        MPI.COMM_WORLD.Barrier()
        end = MPI.Wtime()

        treeSamples = gather_all(treeSamples)
        mcmcLabels = [dt.stats(x, X_test).predict(X_test) for x in treeSamples]
        mcmcAccuracy = [dt.accuracy(y_test, x) for x in mcmcLabels]
        #print("MCMC mean accuracy: ", np.mean(mcmcAccuracy[250:499]))
    if MPI.COMM_WORLD.Get_rank() == 0:
            accuracies.append(np.mean(smcAccuracy))  # replace mean with majority voting
            runtimes.append(end-start)
            print("MCMC mean accuracy: ", accuracies[-1])
            print("MCMC run-time: ", runtimes[-1])
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("MCMC mean of mean accuracies: ", np.mean(accuracies))
        print("MCMC median runtime: ", np.median(runtimes))
except ZeroDivisionError:
    print("MCMC sampling failed due to division by zero")

'''
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

'''