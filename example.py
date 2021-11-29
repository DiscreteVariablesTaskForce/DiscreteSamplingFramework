from discretesampling import spectrum
import math

#Starting dimension 4
startDimension = spectrum.SpectrumDimension(2)

#E.g. Proposal distribution with values and probabilities relative to starting dimentison
q = spectrum.SpectrumDimensionDistribution(startDimension)

q.eval(spectrum.SpectrumDimension(1)) #Returns relevant probabilities from PMF
q.eval(spectrum.SpectrumDimension(2))

q.eval(spectrum.SpectrumDimension(10)) #Returns 0

x = q.sample() #Returns a SpectrumDimension object
x.value #Check the value

## How to use for Metropolis-Hastings

current = spectrum.SpectrumDimension(2)
print("Current dim: " + str(current.value))

forward = spectrum.SpectrumDimensionDistribution(current)

proposed = forward.sample()
print("Proposed dim: " + str(proposed.value))

reverse = spectrum.SpectrumDimensionDistribution(proposed)

#In reality "target" is probably more complicated than this
from discretesampling import discrete
target = discrete.DiscreteVariableDistribution([spectrum.SpectrumDimension(x) for x in [1,2,3,4]], [0.1,0.5,0.2,0.2])


logAcceptanceRatio = target.eval(proposed) - target.eval(current) + reverse.eval(current) - forward.eval(proposed)
print("Log Acceptance ratio: " + str(logAcceptanceRatio))

acceptanceLogProbability = min(0, logAcceptanceRatio)
print("Acceptance log-probability: " + str(acceptanceLogProbability))
print("Acceptance probability: " + str(math.exp(acceptanceLogProbability)))
