import math
import numpy as np
import itertools
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import mpi4py
import joblib

from discretesampling.base.random import RNG
from discretesampling.base.executor import Executor
from discretesampling.base.algorithms.smc_components.effective_sample_size import ess
from discretesampling.base.algorithms.smc_components.resampling import systematic_resampling
from discretesampling.base.algorithms.smc_components.normalisation import normalise

from discretesampling.domain.gaussian_mixture.mix_model_initial_proposal import UnivariateGMMInitialProposal
from discretesampling.domain.gaussian_mixture.mix_model_structure import Gaussian_Mix_Model
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
def RJMCMC(current, t):
    i = 0
    accept = 0
    k = []
    probs = []
    comps = []
    dat = current.Data_Allocation.all_data()
    while i <= t:

        proposed_cont = current.continuous_forward_sample(fixed_beta=False)
        proposed = proposed_cont.discrete_forward_sample()

        print('Starting step {}'.format(i))

        if proposed.last_move == 'split':
            fwd = proposed.split_log_eval(proposed_cont, 0.25)
            back = proposed_cont.merge_log_eval(proposed, 0.25)
            acc_ratio =  (fwd[0] + back[1]) - (fwd[1] + back[0])
            print(f'Split acceptance: {acc_ratio}')
        elif proposed.last_move == 'merge':
            fwd = proposed.merge_log_eval(proposed_cont, 0.25)
            back = proposed_cont.split_log_eval(proposed, 0.25)
            acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
            print(f'Merge acceptance: {acc_ratio}')
        elif proposed.last_move == 'birth':
            fwd = proposed.birth_log_eval(proposed_cont, 0.25)
            back = proposed_cont.death_log_eval(proposed, 0.25)
            acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
           # print(f'Birth acceptance: {acc_ratio}')
        elif proposed.last_move == 'death':
            fwd = proposed.death_log_eval(proposed_cont, 0.25)
            back = proposed_cont.birth_log_eval(proposed, 0.25)
            acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
            #print(f'Death acceptance: {acc_ratio}')
        else:
            acc_ratio = 0

        acc_prob = min([acc_ratio, 0])

        u = np.random.uniform(0, 1)

        if u < math.exp(acc_prob):
            current = proposed
            accept += 1
        else:
            current = proposed_cont


        k.append(current.Gaussian_Mix_Model.k)
        #print('Current length: {}'.format(k[-1]))
        probs.append(current.compute_logprob(dat))
        comps.append(current)
        #print('Current means: {}'.format(current.Gaussian_Mix_Model.means))
        #print('Current vars: {}'.format(current.Gaussian_Mix_Model.vars))
        #print('Current weights: {}'.format(current.Gaussian_Mix_Model.wts))
        #print('Current beta: {}'.format(current.beta))
        i+=1
        #print('Current acceptance ratio: {}'.format(accept / i))

    return k, probs, comps

def straight_SMC(init, N, T):
    t = 0
    n = 0
    particle_path = []
    logwt_path = []
    eff_ss = []

    # Initialise the particles and weights
    front = []
    while n < N:
        front.append(init.get_initial_dist())
        n += 1
    logwts_front = [-math.log(n)] * N

    print(f'Beginning step {t}')
    while t < T:

        print(f'Beginning step {t}')

        # check ESS of sampling front, and resample if necessary
        neff = ess(logwts_front, exec=Executor())
        eff_ss.append(neff)
        print(f'Effective sample size is {neff}')

        if math.log(neff) < math.log(N) - math.log(2):
            print('Resampling')
            front, logwts_front = systematic_resampling(
                front, np.array(logwts_front), rng=RNG(), exec=Executor())

        # Propose new samples from the particle front
        proposed_cont = [(front[i].continuous_forward_sample()) for i in range(N)]
        front = [(proposed_cont[i].discrete_forward_sample()) for i in range(N)]

        # Compute discrete probabilities
        accrat = []
        for i in range(N):
            if front[i].last_move == 'split':
                fwd = front[i].split_log_eval(proposed_cont[i], 0.25)
                back = proposed_cont[i].merge_log_eval(front[i], 0.25)
                acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
                accrat.append(min(0, acc_ratio))
            elif front[i].last_move == 'merge':
                fwd = front[i].merge_log_eval(proposed_cont[i], 0.25)
                back = proposed_cont[i].split_log_eval(front[i], 0.25)
                acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
                accrat.append(min(0, acc_ratio))
            elif front[i].last_move == 'birth':
                fwd = front[i].birth_log_eval(proposed_cont[i], 0.25)
                back = proposed_cont[i].death_log_eval(front[i], 0.25)
                acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
                accrat.append(min(0, acc_ratio))

            elif front[i].last_move == 'death':
                fwd = front[i].death_log_eval(proposed_cont[i], 0.25)
                back = proposed_cont[i].birth_log_eval(front[i], 0.25)
                acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
                accrat.append(min(0, acc_ratio))
            else:
                accrat.append(0)

        # Update and normalise weights
        logwts_front = [logwts_front[i] + accrat[i] for i in range(N)]
        wtsum = logsumexp(logwts_front)
        normwts = logwts_front - wtsum
        for i in range(N):
            logwts_front[i] = normwts[i]

        # retain sample
        particle_path.append(front)
        logwt_path.append(logwts_front)

        t += 1

    return particle_path, logwt_path, eff_ss


def staggered_SMC(init, N, T):
    t = 0
    n = 0
    particle_path = []
    logwt_path = []
    eff_ss = []

    #Initialise the particles and weights
    front = []
    while n<N:
        front.append(init.get_initial_dist())
        n+=1
    logwts_front = [-math.log(n)] * N

    #Initialise the 'staggered front', which remembers which particle to sample from
    staggered_pointer = [0] * (N)

    particle_path.append(front)
    logwt_path.append(logwts_front)

    while t < T:

        print(f'Beginning step {t}')

        # Retrieve the most recently accepted samples from the history
        front = [particle_path[i][j] for i in staggered_pointer for j in range(N)]
        logwts_front = [logwt_path[i][j] for i in staggered_pointer for j in range(N)]

        # Sample new particles
        proposed_cont = [(front[i].continuous_forward_sample()) for i in range(N)]
        front = [(proposed_cont[i].discrete_forward_sample()) for i in range(N)]

        #Compute discrete probabilities
        accrat = []
        for i in range(N):
            if front[i].last_move == 'split':
                fwd = front[i].split_log_eval(proposed_cont[i], 0.25)
                back = proposed_cont[i].merge_log_eval(front[i], 0.25)
                acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
                accrat.append(min(0,acc_ratio))
            elif front[i].last_move == 'merge':
                fwd = front[i].merge_log_eval(proposed_cont[i], 0.25)
                back = proposed_cont[i].split_log_eval(front[i], 0.25)
                acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
                accrat.append(min(0,acc_ratio))
            elif front[i].last_move == 'birth':
                fwd = front[i].birth_log_eval(proposed_cont[i], 0.25)
                back = proposed_cont[i].death_log_eval(front[i], 0.25)
                acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
                accrat.append(min(0,acc_ratio))

            elif front[i].last_move == 'death':
                fwd = front[i].death_log_eval(proposed_cont[i], 0.25)
                back = proposed_cont[i].birth_log_eval(front[i], 0.25)
                acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
                accrat.append(min(0,acc_ratio))
            else:
                accrat.append(0)

        # Update and normalise weights
        logwts_front = [min(logwts_front[i] + accrat[i], logwts_front[i]) for i in range(N)]
        wtsum = logsumexp(logwts_front)
        normwts = logwts_front - wtsum
        for i in range(N):
            logwts_front[i] = normwts[i]

        # check ESS of sample front, and resample if necessary
        neff = ess(logwts_front, exec=Executor())
        eff_ss.append(neff)
        print(f'Effective sample size is {neff}')

        if math.log(neff) < math.log(N) - math.log(2):
            print('Resampling')
            front, logwts_front = systematic_resampling(
                front, np.array(logwts_front), rng=RNG(), exec=Executor())

        #retain sample
        particle_path.append(front)
        logwt_path.append(logwts_front)

        #Update staggered_pointer to reference most recently accepted sample
        u = np.random.uniform(size=N)
        for i in range(len(u)):
            print(accrat[i])
            if u[i] < math.exp(accrat[i]):
                staggered_pointer[i]+=1
        print(f'Last accepted sample at time t={staggered_pointer}')

        t += 1

    return particle_path, logwt_path, eff_ss

def particle_dictionary(parts, wts):

    part_dict = {}
    for i in range(len(parts)):
        k = parts.Gaussian_Mix_Model.k
        if k not in part_dict.keys():
            part_dict[k] = [(i, parts[i], wts[i])]
        else:
            part_dict[k].append((i, parts[i], wts[i]))

    return part_dict

def unpack_particle_dictionary(part_dict):

    sort_all = sorted(itertools.chain(part_dict[i] for i in part_dict), key = lambda x: x[0])

    return [i[1] for i in sort_all], [i[2] for i in sort_all]

def group_by_k(particles, logwts):
    outdict = {}
    for i in range(len(particles)):
        k = particles[i].Gaussian_Mix_Model.k
        if k not in outdict.keys():
            outdict[k] = ([],[])

        outdict[k][0].append(particles[i])
        outdict[k][1].append(logwts[i])

    return dict(sorted(outdict.items()))

def norm_move(pmf, index):

    loc_pmf = [pmf[index-1], pmf[index], pmf[index-1]]
    #print(f'pmf = {loc_pmf}')
    if loc_pmf == [0,0,0]:
        loc_pmf = [0,1,0]
    return np.array(loc_pmf)/sum(loc_pmf)

def weight_informed_SMC(init, N, T):

    #Initialise n particles and compute their weights

    t = 0
    n = 0
    particle_path = []
    logwt_path = []
    eff_ss = [N]

    # Initialise the particles and weights
    front = []
    while n < N:
        front.append(init.get_initial_dist())
        n += 1
    logwts_front = [-math.log(n)] * N

    while t < T:
        print(f'starting step {t}')
        # Compute jump pmf
        sorted_dict = group_by_k(front, logwts_front)
        wtdict = {}
        totprob = 0
        for i in sorted_dict:
            normedwts = normalise(sorted_dict[i][1], exec=Executor())
            locess = ess(normedwts, exec = Executor())
            countprob = sum(np.exp(sorted_dict[i][1]))*(locess/len(normedwts))
            totprob += countprob
            wtdict[i] = countprob
        all_poss = [i for i in range(min(1, min(sorted_dict.keys())), max(sorted_dict.keys()) + 2)]
        std = max((1-totprob)/N, 0)
        allpmf = []
        for i in all_poss:
            if i in wtdict.keys():
                allpmf.append(wtdict[i])
            else:
                allpmf.append(std)

        print(f'pmf= {allpmf} ')
        print(f'length = {len(allpmf)}')

        #discrete sample using normalised pmfs for each component
        eval_probs = [norm_move(allpmf, i.Gaussian_Mix_Model.k) for i in front]
        proposed_cont = [(front[i].continuous_forward_sample()) for i in range(N)]
        front = [proposed_cont[i].discrete_forward_sample(move_pmf = eval_probs[i]) for i in range(N)]

        # Compute discrete probabilities
        accrat = []
        for i in range(N):
            if front[i].last_move == 'split':
                fwd = front[i].split_log_eval(proposed_cont[i], eval_probs[i][2])
                back = proposed_cont[i].merge_log_eval(front[i], eval_probs[i][0])
                acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
                accrat.append(min(0, acc_ratio))
            elif front[i].last_move == 'merge':
                fwd = front[i].merge_log_eval(proposed_cont[i], eval_probs[i][0])
                back = proposed_cont[i].split_log_eval(front[i], eval_probs[i][2])
                acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
                accrat.append(min(0, acc_ratio))
            elif front[i].last_move == 'birth':
                fwd = front[i].birth_log_eval(proposed_cont[i], eval_probs[i][2])
                back = proposed_cont[i].death_log_eval(front[i], eval_probs[i][0])
                acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
                accrat.append(min(0, acc_ratio))
            elif front[i].last_move == 'stick':
                acc_ratio = front[i].eval()-proposed_cont[i].eval()
                accrat.append(min(0, acc_ratio))
            elif front[i].last_move == 'death':
                fwd = front[i].death_log_eval(proposed_cont[i], eval_probs[i][0])
                back = proposed_cont[i].birth_log_eval(front[i], eval_probs[i][2])
                acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
                accrat.append(min(0, acc_ratio))
            else:
                acc_ratio = front[i].eval() - proposed_cont[i].eval()
                accrat.append(min(0, acc_ratio))

        # Update and normalise weights
        logwts_front = [logwts_front[i] + accrat[i] for i in range(N)]
        logwts_front = normalise(logwts_front)

        # retain sample
        particle_path.append(front)
        logwt_path.append(logwts_front)
        #eff_ss.append(N)

        print(f'Average dimension is {np.average([i.Gaussian_Mix_Model.k for i in front])}')

        # check ESS of sample front, and resample if necessary
        neff = ess(logwts_front, exec=Executor())
        eff_ss.append(neff)
        print(f'Effective sample size is {neff}')

        if math.log(neff) < math.log(N) - math.log(2):
            print('Resampling')
            front, logwts_front = systematic_resampling(
                front, np.array(logwts_front), rng=RNG(), exec=Executor())

        t += 1

    return particle_path, logwt_path, eff_ss

if __name__ == "__main__":
    testdir = 'C:/Users/mattb242/Desktop/Projects/reversible_jump/results/toy_model'
    gal_test_data = read_floats_from_file('C:/Users/mattb242/Desktop/Projects/reversible_jump/test_data/enzyme.txt')
    toy_test = Gaussian_Mix_Model([[-8, 1, 0.5], [8, 1, 0.5]])
    toy_test_data = toy_test.sample(100)
    init_dist = UnivariateGMMInitialProposal(2, 0.2, 2, 1, 10, 1, toy_test_data)
    test_SMC = weight_informed_SMC(init_dist, 30, 5000)

    joblib.dump(test_SMC, testdir + '/wt_adjust_toy_30x5000.gz')