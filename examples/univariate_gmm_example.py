import numpy as np
import matplotlib.pyplot as plt


from discretesampling.base.algorithms import DiscreteVariableMCMC
from discretesampling.domain.gaussian_mixture.mix_model_initial_proposal import UnivariateGMMInitialProposal
import discretesampling.domain.gaussian_mixture.mix_model_structure as Ugmm
from discretesampling.domain.gaussian_mixture.mix_model_distribution import UnivariateGMMProposal
from discretesampling.domain.gaussian_mixture.mix_model_target import UnivariateGMMTarget
from discretesampling.domain.gaussian_mixture.util import find_rand

'Generate target test model and sample data'
m = [20.0, 50.0, 100.0]
c = [4, 25, 100]
w = [0.2, 0.5, 0.3]
target = UnivariateGMMTarget(m,c,w)
d = target.sample(1000)

def generate_prior_parameters(g, alpha, la, delta, data):
    m_prior = []
    c_prior = []
    k = np.random.poisson(la)
    zeta = np.median(data)
    kappa = (max(data)-min(data))**-2
    i = 0
    while i<=k:
        m_prior.append(np.random.normal(zeta, kappa**-1))
        c_prior.append(np.random.gamma(2, 10/kappa))
        i+=1
    w_prior = np.random.dirichlet([1]*(k+1))

    return Ugmm.UnivariateGMM(g, alpha, la, delta, m_prior, c_prior, w_prior)

test_prior = generate_prior_parameters(0.2,2,3,1, d)
def gmm_RJMCMC(gmm, iters, d):
    alloc = Ugmm.AllocationStructure(d,gmm)
    curr_dist = gmm.getProposalType(alloc)
    curr_dist.gmm.get_beta_prior(alloc)

    i = 0
    components = []
    mu_1 = []
    mu_2 = []
    mu_3 = []
    while i <= iters:
        prop_dist = curr_dist.sample()
        curr_dist = prop_dist

        components.append(len(prop_dist.gmm.means))
        mu_1.append(min([np.abs(i - 20) for i in curr_dist.gmm.means]))
        mu_2.append(min([np.abs(i - 50) for i in curr_dist.gmm.means]))
        mu_3.append(min([np.abs(i - 100) for i in curr_dist.gmm.means]))
        i += 1

    return prop_dist, components, mu_1, mu_2, mu_3

outprop = gmm_RJMCMC(test_prior, 200, d)

plt.plot(outprop[1])
plt.show()
print('Final means, covariances and weights:')
print(outprop[0].gmm.means)
print(outprop[0].gmm.covs)
print(outprop[0].gmm.compwts)