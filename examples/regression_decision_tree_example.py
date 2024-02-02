from sklearn import datasets
from sklearn.model_selection import train_test_split

from discretesampling.domain import decision_tree as dt
from discretesampling.base.algorithms import DiscreteVariableMCMC, DiscreteVariableSMC

data = datasets.load_diabetes()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)

a = 15
target = dt.RegressionTreeTarget(a)
initialProposal = dt.TreeInitialProposal(X_train, y_train)

# dtMCMC = DiscreteVariableMCMC(dt.Tree, target, initialProposal)
# try:
#     treeSamples = dtMCMC.sample(100)
#
#     mcmcLabels = dt.RegressionStats(treeSamples, X_test).predict(X_test, use_majority=True)
#     mcmcAccuracy = [dt.accuracy_mse(y_test, mcmcLabels)]
#
#     print("MCMC mean MSE: ", (mcmcAccuracy))
# except ZeroDivisionError:
#     print("MCMC sampling failed due to division by zero")


dtSMC = DiscreteVariableSMC(dt.Tree, target, initialProposal)
try:
    treeSMCSamples = dtSMC.sample(10, 50)

    smcLabels = dt.RegressionStats(treeSMCSamples, X_test).predict(X_test, use_majority=True)
    smcAccuracy = [dt.accuracy_mse(y_test, smcLabels)]

    print("SMC mean MSE: ", (smcAccuracy))
except ZeroDivisionError:
    print("SMC sampling failed due to division by zero")
