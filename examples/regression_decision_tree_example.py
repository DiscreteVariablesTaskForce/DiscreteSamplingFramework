from sklearn import datasets
from sklearn.model_selection import train_test_split

from discretesampling.domain import decision_tree as dt
from discretesampling.base.algorithms import DiscreteVariableMCMC, DiscreteVariableSMC
import pandas as pd
from ucimlrepo import fetch_ucirepo
import numpy as np

# df = pd.read_csv(r"C:\Users\efthi\OneDrive\Desktop\PhD\regression_datasets\Walmart.csv")
# df=df.drop(["Date"], axis = 1)
# df = df.dropna()
# y = df.Unemployment
# X = df.drop(['Unemployment'], axis=1)
# X = X.to_numpy()
# y = y.to_numpy()

df = pd.read_csv(r"C:\Users\efthi\OneDrive\Desktop\PhD\regression_datasets\realest.csv")
#df=df.drop(["Date"], axis = 1)
df = df.dropna()
y = df.Price
X = df.drop(['Price'], axis=1)
X = X.to_numpy()
y = y.to_numpy()

#    

# df = pd.read_csv(r"C:\Users\efthi\OneDrive\Desktop\Customer_Churn.csv")
# #df=df.drop(["Date"], axis = 1)
# # df = df.dropna()
# y = df.Target
# X = df.drop(['Target'], axis=1)
# X = X.to_numpy()
# y = y.to_numpy()


total_acc = []

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=i)
    
    a = 100
    target = dt.RegressionTreeTarget(a)
    initialProposal = dt.TreeInitialProposal(X_train, y_train)
    
    # dtMCMC = DiscreteVariableMCMC(dt.Tree, target, initialProposal)
    # try:
    #     treeSamples = dtMCMC.sample(100)
    
    #     mcmcLabels = dt.RegressionStats(treeSamples, X_test).predict(X_test, use_majority=True)
    #     mcmcAccuracy = [dt.accuracy_mse(y_test, mcmcLabels)]
    
    #     print("MCMC mean MSE: ", (mcmcAccuracy))
    # except ZeroDivisionError:
    #     print("MCMC sampling failed due to division by zero")
    
    
    dtSMC = DiscreteVariableSMC(dt.Tree, target, initialProposal)
    try:
        treeSMCSamples, weights = dtSMC.sample(10, 16)
    
        smcLabels = dt.RegressionStats(treeSMCSamples, X_test).predict(X_test, use_majority=True)
        smcAccuracy = [dt.accuracy_mse(y_test, smcLabels)]
        total_acc.append(smcAccuracy)
    
        print("SMC mean MSE: ", (smcAccuracy))
    except ZeroDivisionError:
        print("SMC sampling failed due to division by zero")

print(np.mean(total_acc))

individual_acc = []
smcLabels_individual = dt.RegressionStats(treeSMCSamples, X_test).predict(X_test, use_majority=False)
for lab in smcLabels_individual:
    individual_acc.append(dt.accuracy_mse(y_test, lab))
    