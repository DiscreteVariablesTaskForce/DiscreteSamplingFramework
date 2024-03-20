import numpy as np
from mpi4py import MPI
from sklearn.model_selection import train_test_split
from sklearn import datasets
from discretesampling.base.algorithms import DiscreteVariableSMC
from discretesampling.domain import decision_tree as dt
from discretesampling.base.executor.executor_MPI import Executor_MPI
from discretesampling.base.util import gather_all
import pandas as pd

#data = datasets.load_diabetes()

#X = data.data
#y = data.target

df = pd.read_csv(r"C:\Users\efthi\OneDrive\Desktop\PhD\regression_datasets\realest.csv")
#df=df.drop(["Date"], axis = 1)
df = df.dropna()
y = df.Price
X = df.drop(['Price'], axis=1)
X = X.to_numpy()
y = y.to_numpy()


N = 64
T = 50
seed = 0


exec = Executor_MPI()
try:
    accuracies = []
    for i in range(10):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=i)
        a = 100
        target = dt.RegressionTreeTarget(a)
        initialProposal = dt.TreeInitialProposal(X_train, y_train)
        
        dtSMC = DiscreteVariableSMC(dt.Tree, target, initialProposal, use_optimal_L=False, exec=exec)
        MPI.COMM_WORLD.Barrier()
        start = MPI.Wtime()
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("seed = ", seed)
        treeSMCSamples = dtSMC.sample(T, N, seed)
        MPI.COMM_WORLD.Barrier()
        end = MPI.Wtime()
        if MPI.COMM_WORLD.Get_size() > 1:
            treeSMCSamples = gather_all(treeSMCSamples, exec)

        smcLabels = dt.RegressionStats(treeSMCSamples, X_test).predict(X_test)
        smcAccuracy = dt.accuracy_mse(y_test, smcLabels)
        accuracies.append(smcAccuracy)

        if MPI.COMM_WORLD.Get_rank() == 0:
            print("SMC mean MSE accuracy: ", np.mean(smcAccuracy))
            print("SMC run-time: ", end-start)
    
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("SMC mean of mean accuracies: ", np.mean(accuracies))
except ZeroDivisionError:
    print("SMC sampling failed due to division by zero")
