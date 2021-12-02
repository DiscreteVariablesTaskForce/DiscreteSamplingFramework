import numpy as np
import math
import copy
from scipy.special import logsumexp

class DiscreteVariableSMC():

    def __init__(self, variableType, target, initialProposal):
        self.variableType = variableType
        self.proposalType = variableType.getProposalType()
        self.LKernelType = variableType.getProposalType()
        self.initialProposal = initialProposal
        self.target = target
    
    def sample(self, N, P):
        initialParticles = [self.initialProposal.sample() for p in range(P)]
        
        current_particles = initialParticles
        logWeights = [self.target.eval(p) - self.initialProposal.eval(p) for p in initialParticles]
        logWeights = normaliseLogWeights(logWeights)

        for i in range(N):
                        
            logNeff = calculateNeff(logWeights)
            print("Neff = ", math.exp(logNeff))
            if (logNeff < math.log(P) - math.log(2)):
                print("Resampling...")
                current_particles, logWeights = resample(current_particles, logWeights)
        
            new_particles = copy.deepcopy(current_particles)
            new_logWeights = copy.deepcopy(logWeights)

            for p in range(P):
                
                forward_proposal = self.proposalType(current_particles[p])
                new_particles[p] = forward_proposal.sample()
                
                Lkernel = self.proposalType(new_particles[p])
                
                forward_logprob = forward_proposal.eval(new_particles[p])
                reverse_logprob = Lkernel.eval(current_particles[p])

                current_target_logprob = self.target.eval(current_particles[p])
                new_target_logprob = self.target.eval(new_particles[p])

                new_logWeights[p] = new_target_logprob - current_target_logprob + reverse_logprob - forward_logprob + logWeights[p]
            
            logWeights = normaliseLogWeights(new_logWeights)
            current_particles = new_particles
        
        return current_particles

def calculateNeff(logWeights):
    w = normaliseLogWeights(logWeights)
    log_squared_weights = [2*logW for logW in logWeights]
    return logsumexp(w) - logsumexp(log_squared_weights)

def normaliseLogWeights(logWeights):
    normalisedWeights = logWeights - logsumexp(logWeights)
    return normalisedWeights

def resample(particles, logWeights):
    P = len(particles)
    indexes = range(P)
    new_indexes = np.random.choice(indexes,P,p = [math.exp(logW) for logW in logWeights])
    new_particles = [particles[i] for i in new_indexes]
    new_logWeights = [-math.log(P) for p in range(P)]
    return new_particles, new_logWeights