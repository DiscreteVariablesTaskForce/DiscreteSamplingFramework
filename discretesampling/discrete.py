import numpy as np
import random

class DiscreteVariable:    
    def __init__(self):
        pass

class DiscreteVariableDistribution:    
    def __init__(self, values, probs):
        #Check dims and probs are valid
        assert len(values) == len(probs), "Invalid PMF specified, x and p of different lengths"
        assert sum(probs) == 1.0, "Invalid PMF specified, sum of probabilities != 1.0"
        
        self.x = values
        self.pmf = probs
        self.cmf = np.cumsum(probs)        
        
    def sample(self):
        q = random.random() #random unif(0,1)
        return self.x[np.argmax(self.cmf >= q)]
    
    def eval(self, y):
        try:
            i = self.x.index(y)
            p = self.pmf[i]
        except ValueError:
            print("Warning: value " + str(y) + " not in pmf")
            p = 0
        return p