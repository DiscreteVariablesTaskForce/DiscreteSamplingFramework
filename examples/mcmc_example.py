import numpy
from discretesampling.base.algorithms import DiscreteVariableMCMC
from discretesampling.domain import decision_tree as dt

from sklearn import datasets
from sklearn.model_selection import train_test_split

data = datasets.load_wine()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)

a = 0.01
b = 5

target = dt.TreeTarget(a, b)
initialProposal = dt.TreeInitialProposal(X_train, y_train)

# Create an MCMC sampler on type dt.Tree with target distribution target
dtMCMC = DiscreteVariableMCMC(dt.Tree, target, initialProposal)

try:
    treeSamples = dtMCMC.sample(N=1000)
    mcmc_acc = [dt.accuracy(y_test, dt.stats(x, X_test).predict(X_test)) for x in treeSamples]
    print(numpy.mean(mcmc_acc[501:999]))
except ZeroDivisionError:
    print("MCMC sampling failed due to division by zero")
