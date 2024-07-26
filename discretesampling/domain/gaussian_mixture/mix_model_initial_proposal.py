import sys
sys.path.append('C:/Users/mattb242/Desktop/Projects/reversible_jump/local_code/DiscreteSamplingFramework')

import numpy as np
import pandas as pd

from discretesampling.base.types import DiscreteVariableInitialProposal
from scipy.stats import gamma
from scipy.stats import invgamma
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import dirichlet

from discretesampling.domain.gaussian_mixture.mix_model_structure import Gaussian_Mix_Model
from discretesampling.domain.gaussian_mixture.mix_model_distribution import Data_Allocation
from discretesampling.domain.gaussian_mixture.mix_model_distribution import GMM_Distribution

class UnivariateGMMInitialProposal(DiscreteVariableInitialProposal):

    def __init__(self, la, g, alpha, delta, h_epsilon, k_epsilon, data):
        self.data = data
        self.g = g
        self.la = la
        self.delta = delta
        self.h_epsilon = h_epsilon
        self.k_epsilon = k_epsilon
        self.alpha = alpha
        self.data = data

        self.zeta = np.median(data)
        self.kappa = k_epsilon* (max(data)-min(data))**-2
        self.h = self.h_epsilon*(max(data)-min(data))**-2
        self.beta = gamma.rvs(self.g, scale = self.h)

    def get_initial_dist(self):
        init_comps = []
        k = max(1,poisson.rvs(self.la))
        d = [self.delta]*k
        wt_gen = dirichlet.rvs(d)[0]
        i = 0
        while i < k:
            mu_gen = norm.rvs(self.zeta, 1/self.kappa)
            var_gen = (invgamma.rvs(self.alpha, self.beta))
            init_comps.append([mu_gen, var_gen, wt_gen[i]])
            i+=1

        gmm =  Gaussian_Mix_Model(init_comps)

        alloc = Data_Allocation(gmm.allocate_data(self.data)[0])

        return GMM_Distribution(gmm, alloc, self.la, self.delta, self.alpha, self.g, self.h_epsilon, self.k_epsilon)

