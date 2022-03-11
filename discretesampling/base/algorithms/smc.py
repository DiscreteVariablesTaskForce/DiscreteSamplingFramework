import numpy as np
import math
import copy
from scipy.special import logsumexp


class DiscreteVariableSMC():

    def __init__(self, variableType, target, initialProposal,
                 use_optimal_L=False):
        self.variableType = variableType
        self.proposalType = variableType.getProposalType()
        self.use_optimal_L = use_optimal_L
        if use_optimal_L:
            self.LKernelType = variableType.getOptimalLKernelType()
        else:
            # By default getLKernelType just returns
            # variableType.getProposalType(), the same as the forward_proposal
            self.LKernelType = variableType.getLKernelType()

        self.initialProposal = initialProposal
        self.target = target

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
                    current_particles, logWeights = resample(current_particles,
                                                             logWeights)
                except ValueError as error:
                    raise RuntimeError('Weights do not sum to one, sum = ' +
                                       str(math.exp(logsumexp(logWeights))))\
                                       from error

            new_particles = copy.deepcopy(current_particles)
            new_logWeights = copy.deepcopy(logWeights)

            for p in range(P):

                forward_proposal = self.proposalType(current_particles[p])
                new_particles[p] = forward_proposal.sample()

                if self.use_optimal_L:
                    Lkernel = self.LKernelType()
                    reverse_logprob = Lkernel.eval(new_particles[p],
                                                   current_particles, p)

                else:
                    Lkernel = self.LKernelType(new_particles[p])
                    reverse_logprob = Lkernel.eval(current_particles[p])

                forward_logprob = forward_proposal.eval(new_particles[p])

                current_target_logprob = self.target.eval(current_particles[p])
                new_target_logprob = self.target.eval(new_particles[p])

                new_logWeights[p] = new_target_logprob -\
                    current_target_logprob + reverse_logprob -\
                    forward_logprob + logWeights[p]

            logWeights = normaliseLogWeights(new_logWeights)
            current_particles = new_particles

        return current_particles


def calculateNeff(logWeights):
    tmp = np.array(logWeights)
    non_zero_logWeights = tmp[tmp != -math.inf]
    if (len(non_zero_logWeights) > 0):
        return logsumexp(non_zero_logWeights) -\
               logsumexp(2 * non_zero_logWeights)
    else:
        return -math.inf


def normaliseLogWeights(logWeights):
    tmp = np.array(logWeights)
    non_zero_logWeights = tmp[tmp != -math.inf]
    if (len(non_zero_logWeights) > 0):
        tmp[tmp != -math.inf] = non_zero_logWeights -\
            logsumexp(non_zero_logWeights)
    return list(tmp)


def resample(particles, logWeights):
    P = len(particles)
    indexes = range(P)
    new_indexes = np.random.choice(indexes, P,
                                   p=[math.exp(logW) for logW in logWeights])
    new_particles = [particles[i] for i in new_indexes]
    new_logWeights = [-math.log(P) for p in range(P)]
    return new_particles, new_logWeights
