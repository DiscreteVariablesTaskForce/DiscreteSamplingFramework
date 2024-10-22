from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from discretesampling.domain import decision_tree as dt
from discretesampling.base.algorithms import DiscreteVariableMCMC, DiscreteVariableSMC
import numpy as np

df = pd.read_csv(r"C:\Users\efthi\OneDrive\Desktop\PhD\regression_datasets\machines.csv")
# df['adviser'] = pd.Categorical(df['adviser']).codes
# df['32/60'] = pd.Categorical(df['32/60']).codes
#df = df.rename(columns={df.columns[-1]: 'Target'})

# df['processor_brand'] = pd.Categorical(df['processor_brand']).codes
# df['processor_name'] = pd.Categorical(df['processor_name']).codes
# df['processor_gnrtn'] = pd.Categorical(df['processor_gnrtn']).codes
# df['ram_gb'] = pd.Categorical(df['ram_gb']).codes
# df['ram_type'] = pd.Categorical(df['ram_type']).codes
# df['ssd'] = pd.Categorical(df['ssd']).codes
# df['hdd'] = pd.Categorical(df['hdd']).codes
# df['os'] = pd.Categorical(df['os']).codes
# df['os_bit'] = pd.Categorical(df['os_bit']).codes
# df['graphic_card_gb'] = pd.Categorical(df['graphic_card_gb']).codes
# df['weight'] = pd.Categorical(df['weight']).codes
# df['warranty'] = pd.Categorical(df['warranty']).codes
# df['Touchscreen'] = pd.Categorical(df['Touchscreen']).codes
# df['msoffice'] = pd.Categorical(df['msoffice']).codes
# df['rating'] = pd.Categorical(df['rating']).codes


#df=df.drop(["Date"], axis = 1)
# df=df.drop(["month"], axis = 1)
# df=df.drop(["day"], axis = 1)
df = df.dropna()
y = df.Target
X = df.drop(['Target'], axis=1)
X = X.to_numpy()
y = y.to_numpy()



acc = []
# dtMCMC = DiscreteVariableMCMC(dt.Tree, target, initialProposal)
# try:
#     treeSamples = dtMCMC.sample(500)

#     mcmcLabels = dt.stats(treeSamples, X_test).predict(X_test, use_majority=True)
#     mcmcAccuracy = [dt.accuracy(y_test, mcmcLabels)]
#     print("MCMC mean accuracy: ", (mcmcAccuracy))
# except ZeroDivisionError:
#     print("MCMC sampling failed due to division by zero")

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=i)
    a = 100
    #b = 5
    target = dt.RegressionTreeTarget(a)
    initialProposal = dt.TreeInitialProposal(X_train, y_train)
    dtSMC = DiscreteVariableSMC(dt.Tree, target, initialProposal)
    try:
        treeSMCSamples = dtSMC.sample(100, 100, resampling= "variational")#systematic, knapsack, min_error, variational, min_error_imp
    
        smcLabels = dt.RegressionStats(treeSMCSamples, X_test).predict(X_test, use_majority=True)
        smcAccuracy = [dt.accuracy_mse(y_test, smcLabels)]
        print("SMC mean mse: ", np.mean(smcAccuracy))
        acc.append(smcAccuracy)
    
    except ZeroDivisionError:
        print("SMC sampling failed due to division by zero")
    
print("overall MSE for 10 mc runs is: ", np.mean(acc))