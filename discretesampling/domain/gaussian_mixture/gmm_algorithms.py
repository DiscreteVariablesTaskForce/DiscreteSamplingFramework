import math
import numpy as np

import matplotlib.pyplot as plt

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

        proposed_cont = current.continuous_forward_sample()
        proposed = proposed_cont.discrete_forward_sample()

        print('Starting step {}'.format(i))

        current_eval = proposed_cont.eval()
        proposed_eval = proposed.eval()

        if proposed.last_move == 'split':
            acc_ratio = (proposed_eval - current_eval) + (
                    proposed.split_log_eval(proposed_cont, 0.25)[0] - proposed.split_log_eval(proposed_cont, 0.25)[
                1]) + (proposed_cont.merge_log_eval(proposed, 0.25)[0] - proposed_cont.merge_log_eval(proposed, 0.25)[
                1])
        elif proposed.last_move == 'merge':
            acc_ratio = (proposed_eval - current_eval) + (
                    proposed.merge_log_eval(proposed_cont, 0.25)[0] - proposed.merge_log_eval(proposed_cont, 0.25)[
                1]) + (
                                proposed_cont.split_log_eval(proposed, 0.25)[0] -
                                proposed_cont.split_log_eval(proposed, 0.25)[1])
        elif proposed.last_move == 'birth':
            acc_ratio = (proposed.birth_log_eval(proposed_cont, 0.25)[0] - proposed.birth_log_eval(proposed_cont, 0.25)[
                1]) + (
                                proposed_cont.death_log_eval(proposed, 0.25)[0] -
                                proposed_cont.death_log_eval(proposed, 0.25)[1])
        elif proposed.last_move == 'death':
            acc_ratio = (proposed.death_log_eval(proposed_cont, 0.25)[0] - proposed.death_log_eval(proposed_cont, 0.25)[
                1]) + (
                                proposed_cont.birth_log_eval(proposed, 0.25)[0] -
                                proposed_cont.birth_log_eval(proposed, 0.25)[1])
        else:
            acc_ratio = 0

        acc_prob = min([acc_ratio, 0])

        u = np.random.uniform(0, 1)

        if u < math.exp(acc_prob):
            current = proposed
            accept += 1
            print('{} accepted'.format(current.last_move))
        else:
            current = proposed_cont


        k.append(current.Gaussian_Mix_Model.k)
        print('Current length: {}'.format(k[-1]))
        probs.append(current.compute_logprob(dat))
        comps.append(current)
        print('Current means: {}'.format(current.Gaussian_Mix_Model.means))
        print('Current vars: {}'.format(current.Gaussian_Mix_Model.vars))
        print('Empties: {}'.format(current.Data_Allocation.get_empties()))
        i+=1
        print('Current acceptance ratio: {}'.format(accept / i))

    return k, probs, comps

def staggered_SMC(init, n, T):
    t = 0
    current = []
    while t<=n:
        current.append(init.get_init_dist())
        logwts = [-math.log(n)]*n
        staggered_front = [None]*n

    while t < T:
        #check ESS and resample if necessary


        sample_path = []
        front = []
        # Produce origin particles
        for i in range(n):
            if staggered_front[i] is None:
                front.append(current[i])
            else:
                front.append(staggered_front[i])

        # Propose new samples
        proposed_cont = [(front[i].continuous_forward_sample()) for i in range(n)]
        proposed = [(proposed_cont[i].discrete_forward_sample()) for i in range(n)]
        eval_ratio = [proposed[i].eval() - proposed.cont[i].eval() for i in range(n)]

        #Compute L/q for L=q
        accrat = []
        for i in range(n):
            if proposed[i].last_move == 'split':
                accrat.append(eval_ratio[i] + (
                        proposed[i][1].split_log_eval(proposed_cont[i][1], 0.25)[0] - proposed[i][1].split_log_eval(proposed_cont[i][1], 0.25)[
                    1]) + (proposed_cont[i][1].merge_log_eval(proposed[i][1], 0.25)[0] -
                           proposed_cont[i][1].merge_log_eval(proposed[i][1], 0.25)[
                               1]))
            elif proposed.last_move == 'merge':
                accrat.append(eval_ratio[i] + (
                        proposed[i][1].merge_log_eval(proposed_cont[i][1], 0.25)[0] - proposed[i][1].merge_log_eval(proposed_cont[i][1], 0.25)[
                    1]) + (
                                    proposed_cont[i][1].split_log_eval(proposed[i][1], 0.25)[0] -
                                    proposed_cont[i][1].split_log_eval(proposed[i][1], 0.25)[1]))
            elif proposed.last_move == 'birth':
                accrat.append((proposed[i][1].birth_log_eval(proposed_cont[i][1], 0.25)[0] -
                             proposed[i][1].birth_log_eval(proposed_cont[i][1], 0.25)[
                                 1]) + (
                                    proposed_cont[i][1].death_log_eval(proposed[i][1], 0.25)[0] -
                                    proposed_cont[i][1].death_log_eval(proposed[i][1], 0.25)[1]))
            elif proposed.last_move == 'death':
                accrat.append((proposed[i][1].death_log_eval(proposed_cont[i][1], 0.25)[0] -
                             proposed[i][1].death_log_eval(proposed_cont[i][1], 0.25)[
                                 1]) + (
                                    proposed_cont[i][1].birth_log_eval(proposed[i][1], 0.25)[0] -
                                    proposed_cont[i][1].birth_log_eval(proposed[i][1], 0.25)[1]))
            else:
                accrat.append(0)

        # Update weights
        wts = [proposed[i][0]+accrat[i] for i in range(n)]
        normwts = normalise(wts)
        for i in range(n):
            proposed[i][0] = wts[i]

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