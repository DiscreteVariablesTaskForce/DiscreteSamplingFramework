import multiprocessing
import numpy as np
import math
import copy
from scipy.special import logsumexp
from ...base.random import RandomChoices
from ...base.executor import Executor, Executor_MP

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
        
        #self.exec = Executor()
        self.exec = Executor_MP(self.num_cores)


        if use_optimal_L:
            self.LKernelType = variableType.getOptimalLKernelType()
        else:
            # By default getLKernelType just returns
            # variableType.getProposalType(), the same as the forward_proposal
            self.LKernelType = variableType.getLKernelType()

        self.initialProposal = initialProposal
        self.target = target

    def evolve_particle(self, particle):
        forward_proposal = self.proposalType(particle)
        new_particle = forward_proposal.sample()
        forward_logprob = forward_proposal.eval(new_particle)

        return new_particle, forward_logprob

    def evolve(self, particles):
        new_particles, forward_logprob = zip(*self.exec.map(
            self.evolve_particle,
            particles
        ))

        return list(new_particles), list(forward_logprob)

    def eval_L_particle(self, current_particle, new_particle):
        return self.LKernelType(new_particle).eval(current_particle)

    def evaluate_LKernel(self, current_particles, new_particles):
        P = len(current_particles)
        reverse_logprob = np.zeros(P)

        if self.use_optimal_L:
            Lkernel = self.LKernelType(
                new_particles, current_particles, parallel=self.parallel,
                num_cores=self.num_cores
            )
            reverse_logprob = list(self.exec.map(Lkernel.eval, current_particles))

        else:
            reverse_logprob = list(self.exec.map(self.eval_L_particle, current_particles, new_particles))

        return reverse_logprob

    def update_weights(self, current_particles, new_particles, logWeights,
                       forward_logprob, reverse_logprob):

        def update_weight_particle(current_particle, new_particle, logWeight,
                forward_logprob, reverse_logprob):
            
            current_target_logprob = self.target.eval(current_particle)
            new_target_logprob = self.target.eval(new_particle)
            new_logWeight = (new_target_logprob
                             - current_target_logprob
                             + reverse_logprob
                             - forward_logprob
                             + logWeight)
            return new_logWeight

        new_logWeights = list(self.exec.map(
            update_weight_particle, current_particles, new_particles,
            logWeights, forward_logprob, reverse_logprob
        ))

        return new_logWeights

    def sample(self, N, P):

        initialParticles = [self.initialProposal.sample() for p in range(P)]

        current_particles = initialParticles
        logWeights = [self.target.eval(p) - self.initialProposal.eval(p)
                      for p in initialParticles]
        logWeights = normaliseLogWeights(logWeights)
        for i in range(N):
            logNeff = calculateNeff(logWeights)
            print("Neff = ", math.exp(logNeff))

            #Resample if Neff below threshold
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

            # Sample new particles and calculate forward probabilities
            new_particles, forward_logprob = self.evolve(current_particles)

            # Evaluate L kernel
            reverse_logprob = self.evaluate_LKernel(current_particles, new_particles)

            new_logWeights = self.update_weights(current_particles,
                                                 new_particles, logWeights,
                                                 forward_logprob,
                                                 reverse_logprob)

            logWeights = normaliseLogWeights(new_logWeights)

            current_particles = new_particles

        return current_particles


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
