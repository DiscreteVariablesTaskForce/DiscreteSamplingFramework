import numpy as np
import copy
from discretesampling.base.random import RNG
from discretesampling.domain.gaussian_mixture import mix_model_structure as gmm
from discretesampling.domain.gaussian_mixture.util import find_rand
from discretesampling.domain.gaussian_mixture.mix_model_initial_proposal import UnivariateGMMInitialProposal

import matplotlib.pyplot as plt
import pytest


m_targ = [20.0, 50.0, 100.0]
c_targ = [4, 25, 100]
w_targ = [0.2, 0.3, 0.5]


def sample_multigmm(size, mus, covs, wts):
    sample = []
    wt_cmf = np.cumsum(wts)
    while len(sample) < size:
        q = np.random.uniform(0,1)
        comp = find_rand(wt_cmf, q)
        sample.append(np.random.normal(mus[comp], np.sqrt(covs[comp])))

    return sample

s = sample_multigmm(1000, m_targ, c_targ, w_targ)
zeta = np.median(s)
kappa = 1/(max(s)-min(s))
m_prior = []
c_prior = []
i = 0
while i<=2:
    m_prior.append(np.random.normal(zeta, kappa**-1))
    c_prior.append(np.random.gamma(2, 10/kappa))
    i+=1

w_prior = np.random.dirichlet([1,1,1])




'''
def test_component_count():
    assert test_uni.n_comps == 3
    assert set(test_uni.data_allocations) == {0,1,2}
'''

'''
def test_discrete_update():
    print('First data allocations: {}'.format(test_alloc.current_allocations))
    test_alloc.propose_allocation(test_uni)
    curr_alloc = test_alloc.current_allocations
    curr_count = test_alloc.get_allocation_count(test_uni, curr_alloc)
    print('Initial allocations: {}'.format(curr_count))
    print('With probability {}'.format(test_alloc.current_logprob))
    prop = test_uni.split(test_alloc)
    prop_alloc = test_alloc.proposed_allocations
    prop_count = test_alloc.get_allocation_count(prop, prop_alloc)
    print('Proposed allocation count after birth move: {}'.format(prop_count))
    print('With probability {}'.format(test_alloc.proposed_logprob))
    test_alloc.clear_current_proposal()
    curr_alloc = test_alloc.current_allocations
    curr_count = test_alloc.get_allocation_count(prop, curr_alloc)
    print('After clearance, allocations: {}'.format(curr_count))
    print('With probability {}'.format(test_alloc.current_logprob))
    prop = prop.merge(test_alloc)
    prop_alloc = test_alloc.proposed_allocations
    prop_count = test_alloc.get_allocation_count(prop, prop_alloc)
    print('Proposed allocation count after death move: {}'.format(prop_count))
    print('With probability {}'.format(test_alloc.proposed_logprob))
'''

def test_sweep():
    prop = gmm.UnivariateGMM(0.2, 2, 3, 1, sorted(m_prior), c_prior, w_prior)
    test_alloc = gmm.AllocationStructure(s, prop)
    print('First parameters: {}'.format([prop.means, prop.covs, prop.compwts]))
    i = 0
    while i <20:
        print('iteration {}'.format(i))
        #print('First data allocations: {}'.format(test_alloc.current_allocations))

        'Weight step'
        print('Updating weights')
        print('Initial weights: {}'.format(prop.compwts))
        prop = prop.update_weights(test_alloc)
        print('New weights: {}'.format(prop.compwts))
        curr_alloc = test_alloc.current_allocations
        if test_alloc.proposed_allocations is None:
            curr_count = test_alloc.get_allocation_count(prop, curr_alloc)
        else:
            curr_count = test_alloc.get_allocation_count(prop, prop_alloc)
        print('Allocation count after move:{}'.format(curr_count))


        'Parameter change step'
        print('Initial parameters are {}'.format([prop.means, prop.covs, prop.compwts]))
        prop = prop.update_parameters(test_alloc)
        prop_alloc = test_alloc.proposed_allocations
        curr_count = test_alloc.get_allocation_count(prop, prop_alloc)

        print('Allocation count after move:{}'.format(curr_count))
        print('Updated parameters are {}'.format([prop.means, prop.covs, prop.compwts]))


        'allocation_update step'
        print('Updating allocation')

        test_alloc.propose_allocation(prop)
        curr_alloc = test_alloc.current_allocations
        curr_count = test_alloc.get_allocation_count(prop, curr_alloc)
        prop_alloc = test_alloc.proposed_allocations
        prop_count = test_alloc.get_allocation_count(prop, prop_alloc)
        print('Allocation count before move:{}'.format(curr_count))
        print('Allocation count after move:{}'.format(prop_count))

        'Beta Update Step'
        print('Updating beta')
        print('Old beta: {}'.format(prop.beta))
        prop = prop.update_beta(test_alloc)
        print('New beta: {}'.format(prop.beta))

        'Split/merge'
        print('Split/merge move')
        test_alloc.clear_current_proposal()
        curr_alloc = test_alloc.current_allocations
        curr_count = test_alloc.get_allocation_count(prop, curr_alloc)
        prop_dist = prop.getProposalType()

        prop = prop_dist.dv_sample(prop, test_alloc, 'split_merge')

        prop_alloc = test_alloc.proposed_allocations
        prop_count = test_alloc.get_allocation_count(prop, prop_alloc)
        print('Allocation count before move:{}'.format(curr_count))
        print('Allocation count after move:{}'.format(prop_count))
        print('POst dv parameters are {}'.format([prop.means, prop.covs, prop.compwts]))

        'Split/merge'
        print('Birth/death move')
        test_alloc.clear_current_proposal()
        curr_alloc = test_alloc.current_allocations
        curr_count = test_alloc.get_allocation_count(prop, curr_alloc)
        prop_dist = prop.getProposalType()

        prop = prop_dist.dv_sample(prop, test_alloc, 'birth_death')

        prop_alloc = test_alloc.proposed_allocations
        prop_count = test_alloc.get_allocation_count(prop, prop_alloc)
        print('Allocation count before move:{}'.format(curr_count))
        print('Allocation count after move:{}'.format(prop_count))
        print('POst dv parameters are {}'.format([prop.means, prop.covs, prop.compwts]))



        i+=1

    print('Final parameters: {}'.format(prop.means, prop.covs, prop.compwts))
    print('Final allocation count: {}'.format(prop_count))