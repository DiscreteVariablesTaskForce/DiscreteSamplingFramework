import numpy as np
import math
import matplotlib.pyplot as plt
from mpi4py import MPI

from discretesampling.base.algorithms.smc_components.normalisation import normalise
from discretesampling.domain.gaussian_mixture.mix_model_structure import Gaussian_Mix_Model
from discretesampling.domain.gaussian_mixture.mix_model_initial_proposal import UnivariateGMMInitialProposal
from discretesampling.domain.gaussian_mixture.parallel_gmm_smc import straight_SMC_MPI
from discretesampling.domain.gaussian_mixture.parallel_gmm_smc import wt_informed_RJSMC_MPI
from discretesampling.domain.gaussian_mixture.parallel_gmm_smc import RJMCMC, nRJMCMC

def read_floats_from_file(filepath):
    floats = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                floateval = [float(value) for value in line.split()]
                floats.extend(floateval)
            except ValueError:
                print(f'Non-float detected in line {line}')

    return floats

testdir = 'C:/users/mattb242/Desktop/Projects/reversible_jump/test_data'
file_dir = 'C:/users/mattb242/Desktop/Projects/reversible_jump/results/toy_model'
gal_test_data = read_floats_from_file(testdir + '/galaxy.txt')
toy_test = Gaussian_Mix_Model([[-8, 1, 0.3], [8, 1, 0.7]])
toy_test_data = toy_test.sample(100)
init_dist = UnivariateGMMInitialProposal(3, 0.2, 2, 1, 10, 1, toy_test_data)

rjmcmc_out = nRJMCMC(init_dist, 32000,16000)

x = [i for i in range(len(rjmcmc_out[1]))]
plt.plot(x, rjmcmc_out[1])
plt.ylabel('k')
plt.ylabel('t')
plt.savefig(file_dir + '/toy_rjmcmc_32000_k')
plt.cla()

x = [i for i in range(len(rjmcmc_out[2]))]
plt.plot(x, rjmcmc_out[2])
plt.ylabel('BIC')
plt.ylabel('t')
plt.savefig(file_dir + '/toy_rjmcmc_32000_bic')
plt.cla()

ks = rjmcmc_out[1]
comps = list(set(ks))
comp_probs = {}
avcomp = {}
avbic = {}

for i in comps:

    comp_probs[i] = ks.count(i)/len(ks)
    comp_i = [rjmcmc_out[0][j].Gaussian_Mix_Model.components for j in range(len(ks)) if j == i]
    print(f'For length {i}, complen is {comp_i}')
    bic_i = [rjmcmc_out[2][j] for j in range(len(ks)) if j == i]
    sumav = np.average(np.array(comp_i), axis = 0)
    avcomp[i] = Gaussian_Mix_Model(sumav)
    avbic[i] = np.average(bic_i)

b = min(toy_test_data) - 5
f = max(toy_test_data) + 5
x_new = np.linspace(b, f, 500)
bics = []

for i in avcomp:
    y_new = [avcomp[i].eval(j) for j in x_new]
    plt.plot(x_new,y_new, label = f'k {i}, p = {comp_probs[i]}, BIC = {avbic[i]}')
    plt.hist(toy_test_data, density = True)

plt.legend()
plt.savefig(file_dir + '/toy_rjmcmc_32000_comps')
plt.cla()

min_key = min(avbic, key=avbic.get)
bst = avcomp[min_key]
y_best = [bst.eval(j) for j in x_new]
plt.plot(x_new, y_best, label = f'mu:{bst.means} \n sigma:{bst.vars} \n wt:{bst.wts}')
plt.hist(toy_test_data, density = True)
plt.legend()
plt.savefig(file_dir + '/toy_rjmcmc_32000_best')








