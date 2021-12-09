import numpy as np
import random
import math

class DiscreteVariable:    
    def __init__(self):
        pass
    
    @classmethod
    def getProposalType(self):
        return DiscreteVariableProposal
    
    @classmethod
    def getTargetType(self):
        return DiscreteVariableTarget

    @classmethod
    def getLKernelType(self):
        #Forward proposal
        return self.getProposalType()

    @classmethod
    def getOptimalLKernelType(self):
        return DiscreteVariableOptimalLKernel

class DiscreteVariableProposal:    
    def __init__(self, values, probs):
        #Check dims and probs are valid
        assert len(values) == len(probs), "Invalid PMF specified, x and p of different lengths"
        probs = np.array(probs)
        tolerance = np.sqrt(np.finfo(np.float64).eps)
        assert abs(1 - sum(probs)) < tolerance, "Invalid PMF specified, sum of probabilities !~= 1.0"
        
        self.x = values
        self.pmf = probs
        self.cmf = np.cumsum(probs)        
        
    def sample(self):
        q = random.random() #random unif(0,1)
        return self.x[np.argmax(self.cmf >= q)]
    
    def eval(self, y):
        try:
            i = self.x.index(y)
            logp = math.log(self.pmf[i])
        except ValueError:
            print("Warning: value " + str(y) + " not in pmf")
            logp = -math.inf
        return logp


#Exact same as proposal above
class DiscreteVariableInitialProposal:    
    def __init__(self, values, probs):
        #Check dims and probs are valid
        assert len(values) == len(probs), "Invalid PMF specified, x and p of different lengths"
        probs = np.array(probs)
        tolerance = np.sqrt(np.finfo(np.float64).eps)
        assert abs(1 - sum(probs)) < tolerance, "Invalid PMF specified, sum of probabilities !~= 1.0"
        
        self.x = values
        self.pmf = probs
        self.cmf = np.cumsum(probs)        
        
    def sample(self):
        q = random.random() #random unif(0,1)
        return self.x[np.argmax(self.cmf >= q)]
    
    def eval(self, y):
        try:
            i = self.x.index(y)
            logp = math.log(self.pmf[i])
        except ValueError:
            print("Warning: value " + str(y) + " not in pmf")
            logp = -math.inf
        return logp


class DiscreteVariableTarget:
    def __init__(self):
        pass
    
    def eval(self, x):
        logprob = -math.inf
        return logprob


class DiscreteVariableOptimalLKernel:
    def __init__(self, current_particle, previous_particles):
        self.current_particle = current_particle
        self.previous_particles = previous_particles
        self.proposalType = type(current_particle).getProposalType()

    def eval(self, x):
        logprob = -math.inf
        
        for old_particle in self.previous_particles:
            forward_proposal = self.proposalType(old_particle)
            #Do stuff here
        
        return logprob