import numpy as np
import matplotlib.pyplot as plt


from discretesampling.base.algorithms import DiscreteVariableMCMC
from discretesampling.domain.gaussian_mixture.mix_model_initial_proposal import UnivariateGMMInitialProposal
import discretesampling.domain.gaussian_mixture.mix_model_structure as Ugmm
from discretesampling.domain.gaussian_mixture.mix_model_distribution import UnivariateGMMProposal
from discretesampling.domain.gaussian_mixture.mix_model_target import UnivariateGMMTarget
from discretesampling.domain.gaussian_mixture.util import find_rand
from discretesampling.base.executor import Executor

import copy
import math
from tqdm.auto import tqdm
from discretesampling.base.random import RNG
from discretesampling.base.executor import Executor
from discretesampling.base.algorithms.smc_components.normalisation import normalise
from discretesampling.base.algorithms.smc_components.effective_sample_size import ess
from discretesampling.base.algorithms.smc_components.resampling import systematic_resampling



class DiscreteVariableMCMC_GMM():

    def __init__(self, la, g, alpha, delta, epsilon, target_data):

        self.initialProposal = UnivariateGMMInitialProposal(la, g, alpha, delta, epsilon, target_data)
        self.target_data = target_data

    def sample(self, N, verbose=True):

        current = self.initialProposal.dist

        samples = [current.gmm.getTargetType()]

        display_progress_bar = verbose
        progress_bar = tqdm(total=N, desc="MCMC sampling", disable=not display_progress_bar)

        for i in range(N):
            proposed = current.sample()

            if proposed.gmm.last_move in ['birth', 'death']:
                log_acceptance_ratio = current.eval(current, proposed)
                if log_acceptance_ratio > 0:
                    log_acceptance_ratio = 0
                acceptance_probability = min(1, math.exp(log_acceptance_ratio))
            else:
                acceptance_probability = 1

            q = np.random.uniform(0,1)
            # Accept/Reject
            if (q < acceptance_probability):
                current = proposed
            else:
                # Do nothing
                pass

            samples.append(current.gmm.getTargetType())
            progress_bar.update(1)

        progress_bar.close()
        return samples


def resampler(particles, logwts):
    print('Resampling for weights {}'.format(logwts))
    F = np.cumsum([math.exp(i) for i in logwts])
    u = np.random.uniform(0,1)

    resamps = []
    resamp_logwts = []
    while len(resamps) <= len(logwts):
        k = find_rand(F,u)
        resamps.append(particles[k])
        resamp_logwts.append(logwts[k])
        r = u + 1/len(logwts)
        if r > 1:
            u = r-1
        else:
            u=r

    return resamps, resamp_logwts

class GMM_staggered_SMC():

    def __init__(self, target, initialProposal, sampleSize, exec = Executor()):
        self.target = target
        self.initialProposal = initialProposal
        self.exec = exec

        self.test_data = target.sample(sampleSize)

    def sample(self, Tsmc, N, verbose=True):
        initial_particles = []

        display_progress_bar = verbose
        progress_bar = tqdm(total=Tsmc, desc="SMC Sampling", disable=not display_progress_bar)

        i = 0
        while i <= N:
            g = self.initialProposal.initialise_gmm()
            a = self.initialProposal.initialise_allocation(g)
            initial_particles.append(self.initialProposal.return_initial_distribution(g, a))
            i += 1

        current_particles = np.array([i.sample() for i in initial_particles])
        current_targs = np.array([i.gmm.getTargetType() for i in current_particles])

        logWeights = []
        for i in range(len(initial_particles)):
            logwt = 0
            for j in self.test_data:
                j_eval = self.target.eval([j]) - current_targs[i].eval([j])
                logwt += j_eval
            logWeights.append(logwt)

        logWeights = normalise(np.array(logWeights), Executor())
        ness = ess(np.array(logWeights), Executor())

        if verbose:
            print('Initial normalised weights: {}'.format(logWeights))
            print('Initial parameters: {}'.format([[i.means, i.covs, i.compwts] for i in current_targs]))
            print('Effective sample size: {}'.format(ness))

        t = 0
        all_wts = []
        all_dists = []
        while t <= Tsmc:
            if ness < math.log(50) - math.log(2):
                current_particles, logWeights = resampler(
                    current_particles, logWeights)

            all_dists.append([i.gmm.getTargetType() for i in current_particles])

            new_particles = np.array([i.sample() for i in current_particles])

            log_evals = [current_particles[i].eval(current_particles[i], new_particles[i]) for i in
                         range(len(current_particles))]

            logWeights = np.array([logWeights[i] + log_evals[i] for i in range(len(logWeights))])
            logWeights = normalise(np.array(logWeights), Executor())
            all_wts.append(logWeights)
            if verbose:
                print('Updated weights: {}'.format(logWeights))
            ness = ess(np.array(logWeights), Executor())
            current_particles = np.array(new_particles)

            t += 1
            progress_bar.update(1)

        progress_bar.close()
        return all_wts, all_dists

m = [20.0, 50.0, 100.0]
c = [4, 25, 100]
w = [0.2, 0.5, 0.3]
test_target = UnivariateGMMTarget(m, c, w)
d = test_target.sample(1000)
test_init = UnivariateGMMInitialProposal(3,0.2, 2,1, 10, d)

test_RJMCMC = DiscreteVariableMCMC_GMM(3,0.2, 2, 1, 10, d)
test_SMC = GMM_staggered_SMC(test_target, test_init, 50)


SMC_sampwts, SMC_sampdists = test_SMC.sample(10, 10)

SMC_dip = (0,4,9)
for i in SMC_dip:
    k = np.where(SMC_sampwts[i] == max(SMC_sampwts[i]))[0][0]
    best = SMC_sampdists[i][k]
    print('{}th SMC sample'.format(i))
    print('Best SMC Means: {}'.format(best.means))
    print('Best SMC Vars: {}'.format(best.covs))
    print('Best SMC Wts: {}'.format(best.compwts))

RJMCMC_samps = test_RJMCMC.sample(500)

RJMCMC_dip = (0,99,199,299,399,499)
for i in RJMCMC_dip:
    print('{}th sample'.format(i))
    print('Means {}'.format(RJMCMC_samps[i].means))
    print('Vars {}'.format(RJMCMC_samps[i].covs))
    print('Wts {}'.format(RJMCMC_samps[i].compwts))



