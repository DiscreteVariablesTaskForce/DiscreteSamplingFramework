import numpy as np
import math
import matplotlib.pyplot as plt

from discretesampling.base.algorithms.smc_components.normalisation import normalise
from discretesampling.domain.gaussian_mixture.mix_model_structure import Gaussian_Mix_Model
from discretesampling.domain.gaussian_mixture.mix_model_initial_proposal import UnivariateGMMInitialProposal
from discretesampling.domain.gaussian_mixture.parallel_gmm_smc import straight_SMC_MPI
from discretesampling.domain.gaussian_mixture.parallel_gmm_smc import wt_informed_RJSMC_MPI
from discretesampling.domain.gaussian_mixture.parallel_gmm_smc import RJMCMC

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

def MC_get_all_k(comps):
    compdict = {}
    for i in comps:
        if i.Gaussian_Mix_Model.k in compdict.keys():
            compdict[i.Gaussian_Mix_Model.k].append(i)
        else:
            compdict[i.Gaussian_Mix_Model.k] = [i]

    avcompdict = {}
    countcompdict = {}
    for i in compdict:
        countcompdict[i] = len(compdict[i])
        avcompdict[i] = Gaussian_Mix_Model(np.average(np.array([j.Gaussian_Mix_Model.components for j in compdict[i]]), axis = 1))

    return avcompdict, countcompdict

def get_average_k(parts):
    return np.average([parts[i].Gaussian_Mix_Model.k for i in range(len(parts))])
def get_wt_average_k(parts, wts):
    normwts = np.exp(normalise(np.array(wts)))
    ks = np.array([i.Gaussian_Mix_Model.k for i in parts])
    return np.dot(normwts, ks)

def get_wt_av_mean(parts, wts, index):

    return sum([parts[i].Gaussian_Mix_Model.means[index]*math.exp(wts[i]) for i in range(len(wts))])

def get_wt_av_bic(parts, wts):
    normwts = normalise(np.array(wts))
    bics = np.exp(np.array([i.bic() for i in parts]))
    return np.dot(normwts, bics)

def extract_only_k_components(parts, wts, k):
    k_comps = []
    k_wts = []
    #print(f'wts: {wts}')
    for i in range(len(parts)):
        if parts[i].Gaussian_Mix_Model.k == k:
            k_comps.append(parts[i].Gaussian_Mix_Model.components)
            k_wts.append(wts[i])

    if len(k_wts) > 0:
       k_wts = normalise(np.array(k_wts))

    return k_comps, k_wts

def average_model_components(comps, wts):
    if comps == []:
        return 0
    else:
        av = np.zeros_like(comps[0])

        for i in range(len(wts)):

            av+= math.exp(wts[i])*np.array(comps[i])

    return sorted(av, key = lambda x: x[0])

def extract_all_k(partlist, wtlist, ess, k):
    ness = [20] + ess
    across_comps = []
    filtered_ess = []
    for i in range(len(partlist)):
        #print(f'All weights: {wtlist}')

        k_filter = extract_only_k_components(partlist[i], np.array(wtlist[i]), k)
        if len(k_filter[0]) > 0:
            across_comps.append(average_model_components(k_filter[0], k_filter[1]))
            filtered_ess.append(ness[i])

    prop_ess_norm = np.log(np.array(filtered_ess)/sum(filtered_ess))

    final_average = average_model_components(across_comps, prop_ess_norm)

    return final_average

def all_k_pmf(parts):
    d = {}
    N = len(parts)*len(parts[0])
    for i in parts:
        for j in i:
            k = j.Gaussian_Mix_Model.k
            if k in d.keys():
                d[k] += 1
            else:
                d[k] = 1

    for i in d:
        d[i] = d[i]/N

    return d


testdir = 'C:/Users/mattb242/Desktop/Projects/reversible_jump/results/toy_model'
gal_test_data = read_floats_from_file('C:/Users/mattb242/Desktop/Projects/reversible_jump/test_data/enzyme.txt')
toy_test = Gaussian_Mix_Model([[-8, 1, 0.3], [8, 1, 0.7]])
toy_test_data = toy_test.sample(100)
init_dist = UnivariateGMMInitialProposal(2, 0.2, 2, 1, 1, 10, toy_test_data)

test_out =wt_informed_RJSMC_MPI(init_dist, 32, 1000)

compprob = all_k_pmf(test_out[0])

for i in compprob.keys():
    if compprob[i] > 0.1:

        comps = extract_all_k(test_out[0], test_out[1], test_out[2], i)
        print(comps)
        gmix = Gaussian_Mix_Model(comps)
        b = min(toy_test_data) - 5
        f = max(toy_test_data) + 5
        x = np.linspace(b, f, 500)
        dat = [gmix.eval(j) for j in x]
        bic = gmix.bic(toy_test_data)
        plt.plot(x, dat, label = f'k = {i}, p={round(compprob[i], 3)}, BIC = {round(bic,3)}')
        plt.hist(toy_test_data, bins = 40, density = True)

plt.legend()
plt.savefig(testdir + '/wt_toy_32x200_comps')
plt.cla()


kav = [get_wt_average_k(test_out[0][i], test_out[1][i]) for i in range(1,len(test_out[0]))]
ess = test_out[2]
bicav = [get_wt_av_bic(test_out[0][i], test_out[1][i]) for i in range(1,len(test_out[0]))]

x = [i for i in range(len(kav))]
plt.plot(x, kav)
plt.xlabel('t')
plt.ylabel('average k')
plt.savefig(testdir + '/wt_toy_32x1000_avk')
#plt.show()
plt.cla()

x = [i for i in range(len(kav))]
plt.plot(x, bicav)
plt.xlabel('t')
plt.ylabel('BIC')
plt.savefig(testdir + '/wt_toy_32x1000_bic')
#plt.show()
plt.cla()


xe = [i for i in range(len(test_out[2]))]
plt.plot(xe, test_out[2])
plt.xlabel('t')
plt.ylabel('ESS')
plt.savefig(testdir + '/str_toy_32x1000_ess')
#plt.show()
plt.cla()

def divide_and_calculate_proportions(values, N, k):
    # Ensure that N is not greater than the length of the list
    if N > len(values):
        raise ValueError("N cannot be greater than the length of the list")

    # Calculate the size of each sub-list
    sublist_size = len(values) // N

    # Divide the list into N sub-lists
    sublists = [values[i * sublist_size : (i + 1) * sublist_size] for i in range(N)]

    # Calculate the proportion of values in each sub-list that are above k
    proportions = []
    for sublist in sublists:
        if len(sublist) > 0:  # Avoid division by zero
            proportion = sum(1 for x in sublist if x < k) / len(sublist)
            proportions.append(proportion)
        else:
            proportions.append(0)

    # Handle any remaining elements not included due to integer division
    remaining_elements = values[N * sublist_size:]
    if remaining_elements:
        proportion = sum(1 for x in remaining_elements if x > k) / len(remaining_elements)
        proportions.append(proportion)

    return proportions

resampling_rate = divide_and_calculate_proportions(ess, 20, 16)
print('Resampling rate per 50 steps:')
print(resampling_rate)

xres = [i*50 for i in range(len(resampling_rate))]
plt.plot(xres[1:], resampling_rate[1:])
plt.ylabel('Resampling rate per 50 steps')
plt.xlabel('t')
plt.savefig(testdir + '/str_toy_32x1000_resampling')
plt.cla()