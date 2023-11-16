import numpy as np


from discretesampling.base.algorithms import DiscreteVariableMCMC
from discretesampling.domain.gaussian_mixture.mix_model_initial_proposal import UnivariateGMMInitialProposal
from discretesampling.domain.gaussian_mixture.mix_model_structure import UnivariateGMM
from discretesampling.domain.gaussian_mixture.mix_model_distribution import UnivariateGMMProposal
from discretesampling.domain.gaussian_mixture.mix_model_target import UnivariateGMMTarget
from discretesampling.domain.gaussian_mixture.util import find_rand

m = [20.0, 50.0, 100.0]
c = [4, 25, 100]
w = [0.2, 0.5, 0.3]

def sample_multigmm(size, mus, covs, wts):
    sample = []
    wt_cmf = np.cumsum(wts)
    while len(sample) < size:
        q = np.random.uniform(0,1)
        comp = find_rand(wt_cmf, q)
        sample.append(np.random.normal(mus[comp], np.sqrt(covs[comp])))

    return sample

s = sample_multigmm(1000,m,c,w)

target = UnivariateGMMTarget(m,c,w)

initialProposal = UnivariateGMMInitialProposal(s, la=3, g=0.2,alpha = 2,h_epsilon = 10)
means = initialProposal.means
covs = initialProposal.covs
compwts = initialProposal.compwts

disc_var = UnivariateGMM(means, covs, compwts, s, initialProposal.g, initialProposal.alpha, initialProposal.la, initialProposal.delta)

