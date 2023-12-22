import numpy as np
from scipy.stats import poisson
from scipy.stats import norm
from discretesampling.base.types import DiscreteVariableInitialProposal
import discretesampling.domain.gaussian_mixture.util as gmm_util
import discretesampling.domain.gaussian_mixture.mix_model_structure as mms
import discretesampling.domain.gaussian_mixture.mix_model_distribution as mmd
from scipy.stats import invgamma

class UnivariateGMMInitialProposal(DiscreteVariableInitialProposal):
    def __init__(self, la, g, alpha, delta, h_epsilon, data):
        self.data = data
        self.g = g
        self.la = la
        self.delta = delta
        self.h_epsilon = h_epsilon
        self.alpha = alpha
        self.data = data

        self.zeta = np.median(data)
        self.kappa = (max(data)-min(data))**-2
        self.h = self.kappa*self.h_epsilon
        self.beta = np.random.gamma(self.g,self.h)

        self.gmm = self.initialise_gmm()
        self.alloc = self.initialise_allocation(self.gmm)
        self.dist = self.return_initial_distribution(self.gmm, self.alloc)
        self.target = self.gmm.getTargetType()


    def initialise_gmm(self):

        k = min(1,np.random.poisson(self.la))

        i = 0
        mus = []
        covs = []
        wts = gmm_util.normalise(np.random.dirichlet([self.delta] * (k + 1)))
        while i <= k:
            mus.append(np.random.normal(self.zeta, self.kappa))
            covs.append(invgamma.rvs(self.alpha, self.h_epsilon / self.kappa))
            i += 1

        init_gmm = mms.UnivariateGMM(self.g, self.alpha, self.la, self.delta, mus, covs, wts)
        init_gmm.beta = self.beta

        return init_gmm

    def initialise_allocation(self, gmm):

        return mms.AllocationStructure(self.h_epsilon, self.data, gmm)

    def return_initial_distribution(self, gmm, alloc):

        return gmm.getProposalType(alloc)



