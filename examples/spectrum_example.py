from discretesampling.domain import spectrum
from discretesampling.base.algorithms import DiscreteVariableMCMC, DiscreteVariableSMC

target = spectrum.SpectrumDimensionTarget(10, 3.4)  # NB with mean 10 and variance 3.4^2
initialProposal = spectrum.SpectrumDimensionInitialProposal(50)  # Uniform sampling from 0-50

specMCMC = DiscreteVariableMCMC(spectrum.SpectrumDimension, target, initialProposal)
try:
    samples = specMCMC.sample(1000)

except ZeroDivisionError:
    print("MCMC sampling failed due to division by zero")


specSMC = DiscreteVariableSMC(spectrum.SpectrumDimension, target, initialProposal)
try:
    SMCSamples = specSMC.sample(10, 1000)

except ZeroDivisionError:
    print("SMC sampling failed due to division by zero")
