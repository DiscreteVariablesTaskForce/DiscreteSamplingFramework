import numpy as np
from scipy.stats import poisson
from scipy.stats import norm
from discretesampling.base.types import DiscreteVariableInitialProposal
import discretesampling.domain.gaussian_mixture.util as gmm_util

class UnivariateGMMInitialProposal(DiscreteVariableInitialProposal):
    def __init__(self, data, la, g, h_epsilon, alpha, delta=1):
        self.data = data
        self.g = g
        self.la = la
        self.delta = delta
        self.h_epsilon = h_epsilon
        self.alpha = alpha

        self.zeta = np.median(data)
        self.kappa = (max(data)-min(data))**-2
        self.h = self.kappa*self.h_epsilon
        self.beta = np.random.gamma(self.g,self.h)

        inmeans = []
        incovs = []
        comp_count = poisson.rvs(self.la)
        print('Assigning to {} components'.format(comp_count))
        i = 0
        while i < comp_count:
            inmeans.append(np.random.normal(self.zeta, self.kappa**-0.5))
            incovs.append(np.random.gamma(self.alpha, self.beta)**-2)
            i+=1

        param_sort=np.argsort(inmeans)
        self.means = [inmeans[i] for i in param_sort]
        self.covs = [incovs[i] for i in param_sort]

        dirichlet_parameters = [self.delta]*comp_count
        self.compwts = np.random.dirichlet(dirichlet_parameters)

        self.data_allocations = []
        for i in data:
            alloc_probs = []
            log_prob_alloc = []
            for j in range(len(self.means)):
                logp = -((i - self.means[j]) ** 2) / self.covs[j]
                fac = self.compwts[j] / np.sqrt(self.covs[j])
                alloc_probs.append(np.exp(logp) * fac)
            alloc_cdf = np.cumsum(gmm_util.normalise(alloc_probs))
            q = np.random.uniform(0, 1)
            alloc_index = gmm_util.find_rand(alloc_cdf, q)
            log_prob_alloc.append(alloc_probs[alloc_index])
            self.data_allocations.append(alloc_index)

    def eval(self, s) -> float:
        log_tot_eval = 0
        for j in s:
            for i in range(len(self.compwts)):
                log_tot_eval += np.log(self.compwts[i]) + np.log(self.eval_component(i, j))

        return log_tot_eval

    def eval_component(self, ind, x) -> float:
        # evaluate a single mix component

        return norm.pdf(x, self.means[ind], np.sqrt(self.covs[ind]))

    def sample(self, size=1):
        samp = []
        wt_cdf = np.cumsum(self.compwts)
        q = np.random.uniform(0, 1)
        ind = gmm_util.find_rand(wt_cdf, q)
        samp.append(np.random.normal(self.means, np.sqrt(self.covs)))
        if size == 1:
            return samp[0]
        else:
            return samp

m = [20.0, 50.0, 100.0]
c = [4, 25, 100]
w = [0.2, 0.5, 0.3]

def sample_multigmm(size, mus, covs, wts):
    sample = []
    wt_cmf = np.cumsum(wts)
    while len(sample) < size:
        q = np.random.uniform(0,1)
        comp = gmm_util.find_rand(wt_cmf, q)
        sample.append(np.random.normal(mus[comp], np.sqrt(covs[comp])))

    return sample

s = sample_multigmm(1000, m, c, w)

