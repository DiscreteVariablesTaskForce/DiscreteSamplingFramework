from . import discrete
import numpy as np
import random

class SpectrumDimension(discrete.DiscreteVariable): #SpectrumDimension inherits from DiscreteVariable
    def __init__(self, value):        
        super().__init__()
        self.value = value

    def getDistributionType(self):
        return SpectrumDimensionDistribution
    
    #Are equal if values are equal
    def __eq__(self, other):
        if not isinstance(other, SpectrumDimension):
            return NotImplemented
        
        if self.value != other.value:
            return False

        return True
        

class SpectrumDimensionDistribution(discrete.DiscreteVariableDistribution):#SpectrumDimensionDistribution inherits from DiscreteVariableDistribution
    def __init__(self, startingDimension: SpectrumDimension):
        
        startingValue = startingDimension.value

        if startingValue > 0:
            firstValue = startingValue - 1
        else:
            firstValue = 0
        
        dims = [SpectrumDimension(x) for x in range(firstValue,startingValue+2)]
        numDims = len(dims)
        probs = [1/numDims] * numDims
        
        super().__init__(dims, probs)
        
    