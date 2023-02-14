import numpy
from discretesampling.base.algorithms import DiscreteVariableSMC
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
                                use_optimal_L=True, parallel=True, num_cores=10)

    try:
        treeSamples = dtSMC.sample(10, 500)

        smc_acc = [dt.accuracy(y_test, dt.stats(x, X_test).predict(X_test))
                   for x in treeSamples]
        print(numpy.mean(smc_acc))
    except ZeroDivisionError:
        print("SMC sampling failed due to division by zero")
