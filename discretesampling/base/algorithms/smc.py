import multiprocessing
import numpy as np
import math
import copy
from scipy.special import logsumexp
from ...base.random import RandomChoices


class DiscreteVariableSMC():

    def __init__(self, variableType, target, initialProposal,
                 use_optimal_L=False,
                 parallel=False,
                 num_cores=None):
        self.variableType = variableType
        self.proposalType = variableType.getProposalType()
        self.use_optimal_L = use_optimal_L
        self.parallel = parallel
        self.num_cores = num_cores

        if (self.parallel and (num_cores is None)):
            num_cores = multiprocessing.cpu_count()
            print("WARNING: `parallel=True` but `num_cores` not specified; "
                  + "setting `num_cores = ", num_cores, "`")
            self.num_cores = num_cores

        if use_optimal_L:
            self.LKernelType = variableType.getOptimalLKernelType()
        else:
            # By default getLKernelType just returns
            # variableType.getProposalType(), the same as the forward_proposal
            self.LKernelType = variableType.getLKernelType()

        self.initialProposal = initialProposal
        self.target = target

    def evolve_particle(self, particle):
        new_particle = copy.deepcopy(particle)
        forward_logprob = 0

        forward_proposal = self.proposalType(particle)
        new_particle = forward_proposal.sample()
        forward_logprob = forward_proposal.eval(new_particle)

        return new_particle, forward_logprob

    def evolve(self, particles):
        P = len(particles)
        new_particles = copy.deepcopy(particles)
        forward_logprob = np.zeros(P)

        for p in range(P):
            new_particles[p], forward_logprob[p] = self.evolve_particle(particles[p])

        return new_particles, forward_logprob

    def sample(self, N, P):

        initialParticles = [self.initialProposal.sample() for p in range(P)]

        current_particles = initialParticles
        logWeights = [self.target.eval(p) - self.initialProposal.eval(p)
                      for p in initialParticles]
        logWeights = normaliseLogWeights(logWeights)
        for i in range(N):
            logNeff = calculateNeff(logWeights)
            print("Neff = ", math.exp(logNeff))
            if (logNeff < math.log(P) - math.log(2)):
                print("Resampling...")
                try:
                    current_particles, logWeights = resample(
                        current_particles, logWeights
                    )
                except ValueError as error:
                    raise RuntimeError('Weights do not sum to one, sum = '
                                       + str(math.exp(logsumexp(logWeights)))) \
                                       from error

            new_logWeights = copy.deepcopy(logWeights)

            # Sample new particles and calculate forward probabilities

            new_particles, forward_logprob = self.evolve(current_particles)

            new_logWeights = np.full([P], -math.inf)
            if self.use_optimal_L:
                Lkernel = self.LKernelType(
                    new_particles, current_particles, parallel=self.parallel,
                    num_cores=self.num_cores
                )
            for p in range(P):
                if self.use_optimal_L:
                    reverse_logprob = Lkernel.eval(p)

                else:
                    Lkernel = self.LKernelType(new_particles[p])
                    reverse_logprob = Lkernel.eval(current_particles[p])

                current_target_logprob = self.target.eval(current_particles[p])
                new_target_logprob = self.target.eval(new_particles[p])

                new_logWeights[p] = (new_target_logprob
                                     - current_target_logprob
                                     + reverse_logprob
                                     - forward_logprob[p]
                                     + logWeights[p])

            logWeights = normaliseLogWeights(new_logWeights)

            current_particles = new_particles

        return current_particles


class DiscreteVariableSMC_MP(DiscreteVariableSMC):
    def evolve(self, particles):
        P = len(particles)
        new_particles = copy.deepcopy(particles)
        forward_logprob = np.zeros(P)

        with multiprocessing.Pool() as pool:
            new_particles, forward_logrob = zip(*pool.map(super().evolve_particle, particles))

        return new_particles, forward_logprob


def calculateNeff(logWeights):
    tmp = np.array(logWeights)
    non_zero_logWeights = tmp[tmp != -math.inf]
    if (len(non_zero_logWeights) > 0):
        return (logsumexp(non_zero_logWeights)
                - logsumexp(2 * non_zero_logWeights))
    else:
        return -math.inf


def normaliseLogWeights(logWeights):
    tmp = np.array(logWeights)
    non_zero_logWeights = tmp[tmp != -math.inf]
    if (len(non_zero_logWeights) > 0):
        tmp[tmp != -math.inf] = (non_zero_logWeights
                                 - logsumexp(non_zero_logWeights))
    return list(tmp)


def resample(particles, logWeights):
    P = len(particles)
    indexes = range(P)
    weights = np.zeros(P)
    for i in range(P):
        weights[i] = math.exp(logWeights[i])

    new_indexes = RandomChoices(indexes, weights=weights, k=P).eval()
    new_particles = [particles[i] for i in new_indexes]
    new_logWeights = np.full(len(new_particles), -math.log(P))

    return new_particles, new_logWeights
