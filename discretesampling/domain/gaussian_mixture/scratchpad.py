import numpy as np

import numpy as np
import copy
from discretesampling.base.random import RNG
from discretesampling.domain.gaussian_mixture import mix_model_structure as gmm
from discretesampling.domain.gaussian_mixture.util import find_rand
import matplotlib.pyplot as plt
import pytest

m = [2.0, 5.0, 10.0]
c = [0.1, 0.3, 0.2]
w = [0.2, 0.5, 0.3]

def sample_multigmm(size, mus, covs, wts):
    sample = []
    wt_cmf = np.cumsum(wts)
    while len(sample) < size:
        q = np.random.uniform(0,1)
        comp = find_rand(wt_cmf, q)
        sample.append(np.random.normal(mus[comp], covs[comp]))

    return sample

s = sample_multigmm(1000, m, c, w)

test_uni = gmm.UnivariateGMM(w,m,c,s, 0.2,2,5)

histplots = []
for i in [0,1,2,3,4]:
    oneplot = []
    for j in test_uni.data:
        if test_uni.data_allocations[test_uni.data.index(j)] == i:
            oneplot.append(j)
    histplots.append(oneplot)


for i in range(len(histplots)):
    plt.hist(histplots[i], bins = 50)
plt.legend(['0', '1', '2', '3', '4'])
plt.show()


