import numpy as np
from discretesampling.base.types import DiscreteVariableTarget, DiscreteVariable
from scipy.stats import norm
from discretesampling.domain.gaussian_mixture import util as gmm_util

class UnivariateGMMTarget(DiscreteVariableTarget):

    def __init__(self, means, covs, compwts):
        self.means = means
        self.covs = covs
        self.compwts = compwts
        self.n_comps = len(self.compwts)
        self.indices = [i for i in range(self.n_comps)]

    def eval(self, s) -> float:
        log_tot_eval = 0
        for j in s:
            for i in range(len(self.compwts)):
                log_tot_eval += np.log(self.compwts[i])+np.log(self.eval_component(i,j))

        return log_tot_eval

    def eval_component(self, ind, x) -> float:
        #evaluate a single mix component

        return norm.pdf(x, self.means[ind], np.sqrt(self.covs[ind]))
    def evaluatePrior(self, x: DiscreteVariable, data) -> float:

        pass

    def sample(self, size = 1):
        comp_select = np.cumsum(self.compwts)
        sample = []
        i=0

        while i <= size:
            q = np.random.uniform(0,1)
            index = gmm_util.find_rand(comp_select, q)
            sample.append(np.random.normal(self.means[index], self.covs[index]))
            i+=1

        if size == 1:
            return sample[0]
        else:
            return sample

class MultivariateGMMTarget(DiscreteVariableTarget):

    def __init__(self, compwts, mode_vecs, cov_mats):
        self.compwts = compwts
        self.mode_vecs = mode_vecs
        self.cov_mats = cov_mats
        self.dim = len(mode_vecs[0])
        self.n_modes = len(mode_vecs)

    def eval(self, s) -> float:
        log_tot_eval = 0
        for j in s:
            for i in range(len(self.compwts)):
                log_tot_eval += np.log(self.compwts[i]) + np.log(self.eval_component(i, j))

        return log_tot_eval

    def eval_component(self, ind, x) -> float:
        # evaluate a single mix component

        return norm.pdf(x, self.means[ind], np.sqrt(self.covs[ind]))

    def evaluatePrior(self, x, h) -> float:

        pass

    def eval_component(self, ind, x):

        pass