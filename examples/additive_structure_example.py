import discretesampling.domain.additive_structure as addstruct
from discretesampling.base.algorithms import DiscreteVariableMCMC, DiscreteVariableSMC
import numpy as np
import pandas as pd


def func(x):
    return (x.iloc[:, 0] + x.iloc[:, 1]) * (x.iloc[:, 2] + x.iloc[:, 4]) * x.iloc[:, 3]


n = 1000
x = np.random.uniform(-3, 3, n)
x.shape = (200, 5)
x_train = pd.DataFrame(x)
y_train = func(x_train)
true_str = [[1, 2], [3, 5], [4]]
data = list([x_train, y_train])  # some data defining the target

target = addstruct.AdditiveStructureTarget(data)
initialProposal = addstruct.AdditiveStructureInitialProposal([1, 2, 3, 4, 5])

asMCMC = DiscreteVariableMCMC(addstruct.AdditiveStructure, target,
                              initialProposal)
asSamplesMCMC = asMCMC.sample(50)


asSMC = DiscreteVariableSMC(addstruct.AdditiveStructure, target,
                            initialProposal)
try:
    asSamplesSMC = asSMC.sample(5, 10)
except RuntimeError:
    print("AdditiveStructureTarget is not yet implemented for SMC")


asSMCLkern = DiscreteVariableSMC(addstruct.AdditiveStructure, target, initialProposal, use_optimal_L=False)
asSamplesLk = asSMCLkern.sample(5, 10)


# Comparison between the three methods based on sampling the true structure
mcmc = [x.discrete_set for x in asSamplesMCMC]
lkern = [x.discrete_set for x in asSamplesLk]
smc = [x.discrete_set for x in asSamplesSMC]

mcmc.count(true_str)
lkern.count(true_str)
smc.count(true_str)
