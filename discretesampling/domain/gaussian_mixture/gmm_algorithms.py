import math
import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt

from discretesampling.base.random import RNG
from discretesampling.base.executor import Executor_MPI
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
            print(f'Birth acceptance: {acc_ratio}')
        elif proposed.last_move == 'death':
            fwd = proposed.death_log_eval(proposed_cont, 0.25)
            back = proposed_cont.birth_log_eval(proposed, 0.25)
            acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
            print(f'Death acceptance: {acc_ratio}')
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
        print('Current length: {}'.format(k[-1]))
        probs.append(current.compute_logprob(dat))
        comps.append(current)
        print('Current means: {}'.format(current.Gaussian_Mix_Model.means))
        print('Current vars: {}'.format(current.Gaussian_Mix_Model.vars))
        print('Current weights: {}'.format(current.Gaussian_Mix_Model.wts))
        print('Current beta: {}'.format(current.beta))
        i+=1
        print('Current acceptance ratio: {}'.format(accept / i))

    return k, probs, comps

def staggered_SMC(init, n, T):
    t = 0
    particles = []
    logwts = []


    while t<=n:
        particles.append(init.get_init_dist())
        logwts = [-math.log(n)]*n
        staggered_front = [None]*n
        staggered_logwts = [None]*n

    while t < T:

        sample_path = []
        front = []
        # Produce origin particles
        for i in range(n):
            if staggered_front[i] is None:
                particles.append(current[i])
                logwts.append(logwts[i])
            else:
                particles.append(staggered_front[i])
                logwts.append(staggered_logwts[i])

        # check ESS and resample if necessary
        neff = ess(logwts, exec=Executor())

        if math.log(neff) < math.log(n) - math.log(2):
            particles, logwts = systematic_resampling(
                particles, logwts, rng=RNG(), exec=Executor())




        # Propose new samples
        proposed_cont = [(particles[i].continuous_forward_sample()) for i in range(n)]
        proposed = [(proposed_cont[i].discrete_forward_sample()) for i in range(n)]

        #Compute L/q for L=q
        accrat = []
        for i in range(n):
            if proposed[i].last_move == 'split':
                fwd = proposed[i].split_log_eval(proposed_cont[i], 0.25)
                back = proposed_cont[i].merge_log_eval(proposed[i], 0.25)
                acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
                accrat.append(acc_ratio)
            elif proposed.last_move == 'merge':
                fwd = proposed[i].merge_log_eval(proposed_cont[i], 0.25)
                back = proposed_cont[i].split_log_eval(proposed[i], 0.25)
                acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
                accrat.append(acc_ratio)
            elif proposed.last_move == 'birth':
                fwd = proposed[i].birth_log_eval(proposed_cont[i], 0.25)
                back = proposed_cont[i].death_log_eval(proposed[i], 0.25)
                acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
                accrat.append(acc_ratio)

            elif proposed.last_move == 'death':
                fwd = proposed[i].death_log_eval(proposed_cont[i], 0.25)
                back = proposed_cont[i].birth_log_eval(proposed[i], 0.25)
                acc_ratio = (fwd[0] + back[1]) - (fwd[1] + back[0])
                accrat.append(acc_ratio)
            else:
                accrat.append(0)

        # Update weights
        logwts = [logwts[i] + accrat[i] for i in range(n)]
        wtsum = logsumexp(logwts)
        normwts = logwts - wtsum
        for i in range(n):
            proposed[i][0] = normwts[i]

        # Update accepted or unaccepted
        u = np.random.uniform(size=n)
        staggered_front = np.where(u<accrat, None, proposed)

        current = proposed
        sample_path.append(proposed)

        t += 1

    return sample_path


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