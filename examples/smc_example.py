import numpy
from discretesampling.base.algorithms import DiscreteVariableSMC
from discretesampling.base.executor.executor import Executor
import discretesampling.domain.decision_tree as dt

from sklearn import datasets
from sklearn.model_selection import train_test_split

data = datasets.load_wine()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,
                                                    random_state=5)

a = 0.01
b = 5

target = dt.TreeTarget(a, b)
initialProposal = dt.TreeInitialProposal(X_train, y_train)

# Necessary to use multiprocessing
if __name__ == "__main__":
    dtSMC = DiscreteVariableSMC(dt.Tree, target, initialProposal,
                                use_optimal_L=True, exec=Executor())

    try:
        treeSamples = dtSMC.sample(10, 500)
        smcLabels = dt.stats(treeSamples, X_test).predict(X_test, use_majority=True)
        smc_acc = dt.accuracy(y_test, smcLabels)
        print(numpy.mean(smc_acc))
    except ZeroDivisionError:
        print("SMC sampling failed due to division by zero")
