import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from discretesampling.base.algorithms import DiscreteVariableMCMC, DiscreteVariableSMC
from discretesampling.domain import incremental_decision_tree as idt
from discretesampling.domain.incremental_decision_tree.target import (
    IncrementalTreeTarget, log_poisson)


data = datasets.load_wine()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)


class MatchedPriorTarget(IncrementalTreeTarget):
    """
    The structural prior of decision_tree_example.py, so the two examples can be
    compared. Three differences from IncrementalTreeTarget.log_prior:

      classic:  k*log(lam) - log(e^lam - 1) - log(k!)            on k = #leaves
      default:  m*log(lam) - lam - lgamma(m+1) - log_catalan(k)  on m = #internal

      1. the Poisson is keyed on the internal-node count, not the leaf count
      2. there is an extra Catalan prior on tree shape
      3. min_samples_leaf / max_tree_size truncate the support; the classic
         domain has neither, so both are switched off below

    Only (1) and (2) are handled here. The split prior -- uniform feature and
    uniform threshold per internal node -- is already identical in the two
    domains, so it is left alone. At lam=15 the differing additive constants
    agree to 3dp (-log(e^lam - 1) == -lam), and constants cancel in the MH ratio
    anyway, so this prior is numerically the classic one.

    The LIKELIHOODS remain different and cannot be matched by configuration:
    the classic domain uses a plug-in likelihood (leaf frequencies fitted on the
    training data, then scored against that same data), this one uses the
    Dirichlet-Multinomial marginal.
    """

    def log_prior(self, n_nodes, n_leaves):
        return log_poisson(n_leaves, self.problem.lam)


lam = 15
# Matched to the classic domain, which constrains neither.
min_samples_leaf = 0
max_tree_size = None

ss_prop = 0.25
min_data = 20

problem = idt.IncrementalTreeProblem(X_train, y_train, lam=lam,
                                     min_samples_leaf=min_samples_leaf,
                                     max_tree_size=max_tree_size)
target = MatchedPriorTarget(problem)
initialProposal = idt.IncrementalTreeInitialProposal(problem)

# names = ["MH", "DA", "HINTS"]
names = ["MH"]


def make_proposal(name):
    if name == "MH":
        return idt.IncrementalTreeProposal()
    if name == "DA":
        return idt.DAProposal(target, ss_prop=ss_prop, min_data=min_data)
    return idt.HINTSProposal(target, ss_prop=ss_prop, min_data=min_data)


for name in names:
    idtMCMC = DiscreteVariableMCMC(idt.IncrementalTree, target, initialProposal,
                                   proposal=make_proposal(name))
    try:
        treeSamples = idtMCMC.sample(N=20_000)
        tail = treeSamples[19_000:]
        mcmcLabels = idt.predict(tail, X_test)
        # idt.accuracy returns a fraction; scale it to match dt.accuracy
        mcmc_acc = idt.accuracy(y_test, mcmcLabels) * 100
        print("MCMC-%s acceptance rate:  %.3f (last 1000: %.3f)"
              % (name, idtMCMC.acceptance_rate, idtMCMC.tail_acceptance_rate))
        print("MCMC-%s mean leaves:      %.1f"
              % (name, np.mean([len(t.leaf_idx) for t in tail])))
        print("MCMC-%s mean accuracy:    %.2f%%" % (name, mcmc_acc))
    except ZeroDivisionError:
        print("MCMC-" + name + " sampling failed due to division by zero")


for name in names:
    proposal = make_proposal(name)
    idtSMC = DiscreteVariableSMC(idt.IncrementalTree, target, initialProposal,
                                 proposal=proposal, Lkernel=proposal.lkernel())
    try:
        treeSMCSamples = idtSMC.sample(10, 1000)
        smcLabels = idt.predict(treeSMCSamples, X_test)
        smc_acc = idt.accuracy(y_test, smcLabels)
        print("SMC-" + name + " mean accuracy: ", np.mean(smc_acc))
    except ZeroDivisionError:
        print("SMC-" + name + " sampling failed due to division by zero")
