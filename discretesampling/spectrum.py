from . import discrete
import numpy as np
import random

class SpectrumDimension(discrete.DiscreteVariable): #SpectrumDimension inherits from DiscreteVariable
    def __init__(self, value):        
        super().__init__()
        self.value = value
        

class SpectrumDimensionDistribution(discrete.DiscreteVariableDisribution):#SpectrumDimensionDistribution inherits from DiscreteVariableDistribution
    def __init__(self, dims, probs):        
        super().__init__()
        
        #Check dims and probs are valid
        assert len(dims) == len(probs), "Invalid PMF specified, x and p of different lengths"
        assert sum(probs) == 1.0, "Invalid PMF specified, sum of probabilities != 1.0"
        
        self.x = dims
        self.pmf = probs
        self.cmf = np.cumsum(probs)        

    def sample(self):
        q = random.random() #random unif(0,1)
        d = self.x[np.argmax(self.cmf >= q)]
        x = SpectrumDimension(d)
        return x
        
    def eval(self, y):
        try:
            i = self.x.index(y.value)
            p = self.pmf[i]
        except ValueError:
            print("Warning: value " + str(y.value) + " not in pmf")
            p = 0
        return p