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

    def sample(self, N, P):

        initialParticles = [self.initialProposal.sample() for p in range(P)]

        current_particles = initialParticles
        logWeights = [self.target.eval(p) - self.initialProposal.eval(p)
                      for p in initialParticles]
        logWeights = normaliseLogWeights(logWeights)
        for i in range(N):
            current_particles_for_l_kernel = []
            new_particles_for_l_kernel = []
            forward_logprob_idx = []
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

            new_particles = copy.deepcopy(current_particles)
            new_logWeights = copy.deepcopy(logWeights)

            forward_logprob = np.zeros(len(current_particles))
            

            # Sample new particles and calculate forward probabilities
            for p in range(P):
                forward_proposal = self.proposalType(current_particles[p])
                new_particles[p] = forward_proposal.sample()
                forward_logprob[p] = forward_proposal.eval(new_particles[p])
                
            #####new code
            if type(np.where(forward_logprob == np.min(forward_logprob))) == int: 
                current_particles_for_l_kernel.append(current_particles[np.where(forward_logprob == np.min(forward_logprob))])
                new_particles_for_l_kernel.append(new_particles[np.where(forward_logprob == np.min(forward_logprob))])
                forward_logprob_idx.append(forward_logprob[np.where(forward_logprob == np.min(forward_logprob))])
            else:
                current_particles_for_l_kernel.append(current_particles[np.where(forward_logprob == np.min(forward_logprob))[0][0]])
                new_particles_for_l_kernel.append(new_particles[np.where(forward_logprob == np.min(forward_logprob))[0][0]])
                forward_logprob_idx.append(forward_logprob[np.where(forward_logprob == np.min(forward_logprob))[0][0]])
                
            median = np.median(forward_logprob)
            idx = (np.abs(forward_logprob - median)).argmin()
            current_particles_for_l_kernel.append(current_particles[idx])
            new_particles_for_l_kernel.append(new_particles[idx])
            forward_logprob_idx.append(forward_logprob[idx])
            
            if type(np.where(forward_logprob == np.max(forward_logprob))) == int: 
                current_particles_for_l_kernel.append(current_particles[np.where(forward_logprob == np.max(forward_logprob))])
                new_particles_for_l_kernel.append(new_particles[np.where(forward_logprob == np.max(forward_logprob))])
                forward_logprob_idx.append(forward_logprob[np.where(forward_logprob == np.max(forward_logprob))])
            else:
                current_particles_for_l_kernel.append(current_particles[np.where(forward_logprob == np.max(forward_logprob))[0][0]])
                new_particles_for_l_kernel.append(new_particles[np.where(forward_logprob == np.min(forward_logprob))[0][0]])
                forward_logprob_idx.append(forward_logprob[np.where(forward_logprob == np.max(forward_logprob))[0][0]])
            ###new code                      
            
            
                                  

            
            new_logWeights = np.full([P], -math.inf)
            if self.use_optimal_L:
                Lkernel = self.LKernelType(#changed the arguments when we call this functions
                    new_particles_for_l_kernel, current_particles_for_l_kernel, parallel=self.parallel,
                    num_cores=self.num_cores
                )
            for p in range(len(new_particles_for_l_kernel)):#changed that from reange(P) to len(new_particles_for_l_kernel)
                if self.use_optimal_L:
                    reverse_logprob = Lkernel.eval(p)
                
                else:
                    Lkernel = self.LKernelType(new_particles_for_l_kernel[p])#changed the arguments when we call this functions
                    reverse_logprob = Lkernel.eval(current_particles_for_l_kernel[p])#changed the arguments when we call this functions
            
                
            for p in range(P):
                current_target_logprob = self.target.eval(current_particles[p])
                new_target_logprob = self.target.eval(new_particles[p])
                idx = (np.abs(forward_logprob_idx - forward_logprob[p])).argmin()
                new_logWeights[p] = (new_target_logprob
                                     - current_target_logprob
                                     + reverse_logprob
                                     - forward_logprob[idx]
                                     + logWeights[p])

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
