import numpy as np
import copy
from discretesampling.base.random import RNG
from discretesampling.domain.gaussian_mixture import mix_model_structure as gmm
from discretesampling.domain.gaussian_mixture.util import find_rand
from discretesampling.domain.gaussian_mixture.mix_model_initial_proposal import UnivariateGMMInitialProposal

import matplotlib.pyplot as plt
import pytest




def test_full_sweep():
    m_targ = [20.0, 50.0, 100.0]
    c_targ = [4, 25, 100]
    w_targ = [0.2, 0.3, 0.5]

    def sample_multigmm(size, mus, covs, wts):
        sample = []
        wt_cmf = np.cumsum(wts)
        while len(sample) < size:
            q = np.random.uniform(0, 1)
            comp = find_rand(wt_cmf, q)
            sample.append(np.random.normal(mus[comp], np.sqrt(covs[comp])))

        return sample

    s = sample_multigmm(1000, m_targ, c_targ, w_targ)
    zeta = np.median(s)
    kappa = 1 / (max(s) - min(s))
    m_prior = []
    c_prior = []
    k = np.random.poisson(3)
    i = 0
    while i <= k:
        m_prior.append(np.random.normal(zeta, kappa ** -1))
        c_prior.append(np.random.gamma(2, 10 / kappa))
        i += 1

    w_prior = np.random.dirichlet([1] * (k + 1))
    prior_structure = gmm.UnivariateGMM(0.2, 2, 3, 1, sorted(m_prior), c_prior, w_prior)
    test_alloc = gmm.AllocationStructure(s, prior_structure)
    curr_dist = prior_structure.getProposalType(test_alloc)
    curr_dist.gmm.get_beta_prior(test_alloc)

    i = 0
    components = []
    mu_1 = []
    mu_2 = []
    mu_3 = []
    while i <= 1000:
        print(max(curr_dist.gmm.covs))
        prop_dist = curr_dist.sample()

        print('Proposed covariances: {}'.format(prop_dist.gmm.covs))
        print('Proposed means: {}'.format(prop_dist.gmm.means))
        print('Proposed weights:{}'.format(prop_dist.gmm.compwts))

        if len(prop_dist.gmm.means) < 3:
            mu1_diff = min([np.abs(i-20) for i in prop_dist.gmm.means])
            mu2_diff = min([np.abs(i-50) for i in prop_dist.gmm.means])
            mu3_diff = min([np.abs(i - 100) for i in prop_dist.gmm.means])
        else:
            wtsort = list(np.argsort(prop_dist.gmm.compwts))
            end = len(wtsort)
            bigmeans = [prop_dist.gmm.means[wtsort.index(end-1)], prop_dist.gmm.means[wtsort.index(end-2)], prop_dist.gmm.means[wtsort.index(end-3)]]
            mu1_diff = min([np.abs(i - 20) for i in bigmeans])
            mu2_diff = min([np.abs(i - 50) for i in bigmeans])
            mu3_diff = min([np.abs(i - 100) for i in bigmeans])


        #print('Log probability evaluation = {} for move {}, which was a {}'.format(prob, i, prop_dist.gmm.last_move))

        curr_dist = prop_dist

        components.append(len(prop_dist.gmm.means))
        mu_1.append(mu1_diff)
        mu_2.append(mu2_diff)
        mu_3.append(mu3_diff)
        i+=1

    plt.plot(mu_1, label = 'mu_1')
    plt.plot(mu_2, label='mu_2')
    plt.plot(mu_3, label='mu_3')
    plt.legend()
    plt.show()

def test_sweep_SMC():

    pass

