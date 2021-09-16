from discretesampling import spectrum

#E.g. Proposal distribution with values and probabilities
q = spectrum.SpectrumDimensionDistribution([1,2,3], [0.1,0.5,0.4])

q.eval(spectrum.SpectrumDimension(1)) #Returns relevant probabilities from PMF
q.eval(spectrum.SpectrumDimension(2))

q.eval(spectrum.SpectrumDimension(10)) #Returns 0

x = q.sample() #Returns a SpectrumDimension object
x.value #Check the value

## How to use for Metropolis-Hastings

current = spectrum.SpectrumDimension(2)
print("Current dim: " + str(current.value))

forward = spectrum.SpectrumDimensionDistribution([1,2,3], [0.1,0.5,0.4])

proposed = spectrum.SpectrumDimension(3) #let's pretend we sampled it with proposed = forward.sample()
print("Proposed dim: " + str(current.value))

reverse = spectrum.SpectrumDimensionDistribution([2,3,4], [0.3,0.4,0.3])
#We could write an extra function which automatically generates the proposal distributions
#e.g. based on a graph/matrix transition probabilities

#In reality "target" is probably more complicated than this
target = spectrum.SpectrumDimensionDistribution([1,2,3,4], [0.1,0.5,0.2,0.2])

acceptanceRatio = target.eval(proposed)/ target.eval(current) * reverse.eval(current) / forward.eval(proposed)
print("Acceptance ratio: " + str(acceptanceRatio))

acceptanceProbability = min(1, acceptanceRatio)
print("Acceptance probability: " + str(acceptanceProbability))
