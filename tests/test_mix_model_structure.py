import numpy as np
import math
import copy
from discretesampling.base.random import RNG
from discretesampling.domain.gaussian_mixture import mix_model_structure as gmm
from discretesampling.domain.gaussian_mixture.util import find_rand
from discretesampling.domain.gaussian_mixture.mix_model_initial_proposal import UnivariateGMMInitialProposal
from discretesampling.domain.gaussian_mixture.mix_model_target import UnivariateGMMTarget

from discretesampling.base.algorithms.smc_components.effective_sample_size import ess
from discretesampling.base.algorithms.smc_components.resampling import systematic_resampling
from discretesampling.base.algorithms.smc_components.normalisation import normalise

from discretesampling.base.executor import Executor
from discretesampling.base.random import RNG


import matplotlib.pyplot as plt
import pytest

'''
def test_full_sweep():
    m_targ = [20.0, 50.0, 100.0]
    c_targ = [4, 25, 100]
    w_targ = [0.2, 0.3, 0.5]
    x = np.linspace(0, 150, 1000)

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
    target_structure = gmm.UnivariateGMM(0.2, 2, 3, 1, m_targ, c_targ, w_targ)


    test_alloc = gmm.AllocationStructure(s, prior_structure)
    curr_dist = prior_structure.getProposalType(test_alloc)
    pr = [curr_dist.eval_at_x(i) for i in x]
    targ_dist = target_structure.getProposalType(test_alloc)
    t = [targ_dist.eval_at_x(i) for i in x]
    curr_dist.gmm.get_beta_prior(test_alloc)

    i = 0
    components = []
    mu_1 = []
    mu_2 = []
    mu_3 = []
    prop_plots = {}
    while i <= 1000:
        print(max(curr_dist.gmm.covs))
        prop_dist = curr_dist.sample()

        print('Proposed covariances: {}'.format(prop_dist.gmm.covs))
        print('Proposed means: {}'.format(prop_dist.gmm.means))
        print('Proposed weights:{}'.format(prop_dist.gmm.compwts))

        #print('Log probability evaluation = {} for move {}, which was a {}'.format(prob, i, prop_dist.gmm.last_move))

        curr_dist = prop_dist
        if i % 500 == 0:
            prop_plots[i] = [prop_dist.eval_at_x(i) for i in x]
        i+=1

    p = [prop_dist.eval_at_x(i) for i in x]
    plt.plot(x, t, label = 'Target') 
    plt.plot(x, pr, label = 'Prior', linestyle = 'dashed')
    #plt.plot(x, p, label = 't=200', linestyle = 'dashed')

    for i in prop_plots:
        plt.plot(x, prop_plots[i], label = 't = {}'.format(i), linestyle = 'dashed')

    plt.title('RJMCMC Convergence - Toy Model')
    plt.legend()
    plt.show()


'''
def resampler(particles, logwts):
    print('Resampling for weights {}'.format(logwts))
    F = np.cumsum([math.exp(i) for i in logwts])
    u = np.random.uniform(0,1)

    resamps = []
    resamp_logwts = []
    while len(resamps) <= len(logwts):
        k = find_rand(F,u)
        resamps.append(particles[k])
        resamp_logwts.append(logwts[k])
        r = u + 1/len(logwts)
        if r > 1:
            u = r-1
        else:
            u=r

    return resamps, resamp_logwts

def test_sweep_SMC():
    ps = 10
    stps = 100
    m_targ = [20.0, 50.0, 100.0]
    c_targ = [4, 25, 100]
    w_targ = [0.2, 0.3, 0.5]
    targ_dist = UnivariateGMMTarget(m_targ, c_targ, w_targ)
    test_data = targ_dist.sample(50)
    x = np.linspace(0, 150, 1000)


    InitialProposal = UnivariateGMMInitialProposal(3,0.2, 2,1, 10, test_data)
    initial_particles = []

    i = 0
    while i <= ps:
        g = InitialProposal.initialise_gmm()
        a = InitialProposal.initialise_allocation(g)
        initial_particles.append(InitialProposal.return_initial_distribution(g, a))
        i+=1

    current_particles = np.array([i.sample() for i in initial_particles])
    #print('Covariances: {}'.format([i.gmm.covs for i in initial_particles]))
    current_targs = np.array([i.gmm.getTargetType() for i in current_particles])

    logWeights = []
    for i in range(len(initial_particles)):
        logwt = 0
        for j in test_data:
            j_eval = targ_dist.eval([j]) - current_targs[i].eval([j])
            logwt += j_eval
        logWeights.append(logwt)

    logWeights = normalise(np.array(logWeights), Executor())
    ness = ess(np.array(logWeights), Executor())


    print('Initial normalised weights: {}'.format(logWeights))
    print('Initial parameters: {}'.format([[i.means, i.covs, i.compwts] for i in current_targs]))
    print('Effective sample size: {}'.format(ness))

    t = 0
    while t <= stps:
        if ness < math.log(50) - math.log(2):
                current_particles, logWeights = resampler(
                    current_particles, logWeights)

        new_particles = np.array([i.sample() for i in current_particles])

        log_evals = [current_particles[i].eval(current_particles[i], new_particles[i]) for i in range(len(current_particles))]

        logWeights = np.array([logWeights[i] + log_evals[i] for i in range(len(logWeights))])
        logWeights = normalise(np.array(logWeights), Executor())
        print('Updated weights: {}'.format(logWeights))
        ness = ess(np.array(logWeights), Executor())
        current_particles = np.array(new_particles)

        t+=1

    print('Final weights: {}'.format(logWeights))
    print('Final sample size: {}'.format(ness))
    print('Final parameters: {}'.format([[i.gmm.means, i.gmm.covs, i.gmm.compwts] for i in current_particles]))
    k = np.where(logWeights == max(logWeights))[0]
    best = [current_particles[j] for j in k]
    print('Best parameters: {}'.format([[i.gmm.means, i.gmm.covs, i.gmm.compwts] for i in best]))





