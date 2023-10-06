import numpy as np
from mpi4py import MPI
from sklearn.model_selection import train_test_split
from sklearn import datasets
from discretesampling.base.algorithms import DiscreteVariableSMC
from discretesampling.domain import decision_tree as dt
from discretesampling.base.executor.executor_MPI import Executor_MPI
from discretesampling.base.util import gather_all

data = datasets.load_wine()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)

a = 15
target = dt.TreeTarget(a)
initialProposal = dt.TreeInitialProposal(X_train, y_train)
proposal = dt.TreeProposal()
N = 1 << 10
T = 10
seed = 0

exec = Executor_MPI()

dtSMC = DiscreteVariableSMC(dt.Tree, target, initialProposal, proposal=proposal, use_optimal_L=False, exec=exec)
try:
    MPI.COMM_WORLD.Barrier()
    start = MPI.Wtime()

    if MPI.COMM_WORLD.Get_rank() == 0:
        print("seed = ", seed)
    treeSMCSamples = dtSMC.sample(T, N, seed)

    MPI.COMM_WORLD.Barrier()
    end = MPI.Wtime()

    if MPI.COMM_WORLD.Get_size() > 1:
        treeSMCSamples = gather_all(treeSMCSamples, exec)

    smcLabels = dt.stats(treeSMCSamples, X_test).predict(X_test)
    smcAccuracy = dt.accuracy(y_test, smcLabels)

    if MPI.COMM_WORLD.Get_rank() == 0:
        print("SMC mean accuracy: ", np.mean(smcAccuracy))
        print("SMC run-time: ", end-start)
except ZeroDivisionError:
    print("SMC sampling failed due to division by zero")
