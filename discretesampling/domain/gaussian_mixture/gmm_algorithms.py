import math
import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt

from discretesampling.base.random import RNG
from discretesampling.base.executor import Executor
from discretesampling.base.algorithms.smc_components.effective_sample_size import ess
from discretesampling.base.algorithms.smc_components.resampling import systematic_resampling


from discretesampling.domain.gaussian_mixture.mix_model_initial_proposal import UnivariateGMMInitialProposal
from discretesampling.domain.gaussian_mixture.mix_model_structure import Gaussian_Mix_Model

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

    return particle_path, logwt_path


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


def weight_informed_SMC(init, n, t):

    #Initialise n particles and compute their weights

    i = 0
    while i < t:
        #Compute within-model ESS

        #Resample if necessary

        #Compute jump pmf

        #Assign jump probabilities to each particle and compute next move

        #Propose new particles

        #Update weights

        #Normalise weight

        i+=1

    pass