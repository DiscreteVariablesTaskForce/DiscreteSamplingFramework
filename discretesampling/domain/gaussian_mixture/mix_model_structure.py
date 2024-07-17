import sys
sys.path.append('C:/Users/mattb242/Desktop/Projects/reversible_jump/local_code/DiscreteSamplingFramework')

import numpy as np
from scipy.stats import norm

from discretesampling.domain.gaussian_mixture import util

class Gaussian_Mix_Model:

    def __init__(self, components):

        self.components = components
        self.k = len(self.components)

        self.means = np.array([i[0] for i in self.components])
        self.vars = np.array([i[1] for i in self.components])
        self.wts = np.array([i[2] for i in self.components])


    def normalise_weights(self):
        normed = self.wts/sum(self.wts)
        for i in range(len(normed)):
            self.components[i][2] = normed[i]
    def eval(self, x):

        return sum([self.wts[i]*norm.pdf(x, self.means[i], np.sqrt(self.vars[i])) for i in range(self.k)])

    def sample(self, size = 1):
        samp = []
        while len(samp) <= size:
            comp = util.assign_from_pmf(self.wts)
            samp.append(norm.rvs(self.means[comp], np.sqrt(self.vars[comp])))

        if len(samp) == 1:
            return samp[0]
        else:
            return samp

    def allocate_data(self, data):
        log_palloc = 0
        datdict = {i:[] for i in range(self.k)}
        for i in data:
            pmf = [self.wts[j]*norm.pdf(i, self.means[j], np.sqrt(self.vars[j])) for j in range(self.k)]
            norm_pmf = np.exp(pmf-util.logsumexp(pmf))
            assign_index = util.assign_from_pmf(norm_pmf)
            log_palloc += norm.logpdf(i, self.means[assign_index], np.sqrt(self.vars[assign_index]))
            datdict[assign_index].append(i)

        return datdict, log_palloc

    def insert_blank_component(self, index):

        pass


