import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from discretesampling.base.algorithms import DiscreteVariableMCMC, DiscreteVariableSMC
from discretesampling.domain import incremental_decision_tree as idt


data = datasets.load_wine()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)

lam = 5
min_samples_leaf = 5
max_tree_size = 8

ss_prop = 0.25
min_data = 20

problem = idt.IncrementalTreeProblem(X_train, y_train, lam=lam,
                                     min_samples_leaf=min_samples_leaf,
                                     max_tree_size=max_tree_size)
target = idt.IncrementalTreeTarget(problem)
initialProposal = idt.IncrementalTreeInitialProposal(problem)


def make_proposal(name):
    if name == "MH":
        return idt.IncrementalTreeProposal()
    if name == "DA":
        return idt.DAProposal(target, ss_prop=ss_prop, min_data=min_data)
    return idt.HINTSProposal(target, ss_prop=ss_prop, min_data=min_data)


for name in ("MH", "DA", "HINTS"):
    idtMCMC = DiscreteVariableMCMC(idt.IncrementalTree, target, initialProposal,
                                   proposal=make_proposal(name))
    try:
        treeSamples = idtMCMC.sample(N=500)
        mcmcLabels = idt.predict(treeSamples[250:], X_test)
        mcmc_acc = idt.accuracy(y_test, mcmcLabels)
        print("MCMC-" + name + " mean accuracy: ", np.mean(mcmc_acc))
    except ZeroDivisionError:
        print("MCMC-" + name + " sampling failed due to division by zero")


for name in ("MH", "DA", "HINTS"):
    proposal = make_proposal(name)
    idtSMC = DiscreteVariableSMC(idt.IncrementalTree, target, initialProposal,
                                 proposal=proposal, Lkernel=proposal.lkernel())
    try:
        treeSMCSamples = idtSMC.sample(50, 64)
        smcLabels = idt.predict(treeSMCSamples, X_test)
        smc_acc = idt.accuracy(y_test, smcLabels)
        print("SMC-" + name + " mean accuracy: ", np.mean(smc_acc))
    except ZeroDivisionError:
        print("SMC-" + name + " sampling failed due to division by zero")
