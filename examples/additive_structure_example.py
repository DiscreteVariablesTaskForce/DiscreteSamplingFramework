import discretesampling.domain.additive_structure as addstruct
from discretesampling.base.algorithms import DiscreteVariableMCMC, DiscreteVariableSMC

data = []  # some data defining the target
target = addstruct.AdditiveStructureTarget(data)
initialProposal = addstruct.AdditiveStructureInitialProposal([1, 2, 3, 4, 5])

asMCMC = DiscreteVariableMCMC(addstruct.AdditiveStructure, target,
                              initialProposal)
asSamples = asMCMC.sample(1000)

asSMC = DiscreteVariableSMC(addstruct.AdditiveStructure, target,
                            initialProposal)
try:
    asSamples = asSMC.sample(N=10, P=1000)
except RuntimeError:
    print("AdditiveStructureTarget is not yet implemented for SMC")
