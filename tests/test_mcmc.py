import pytest
from discretesampling.base.algorithms import DiscreteVariableMCMC
from discretesampling.domain import spectrum


@pytest.mark.parametrize(
    "seed,T,expected",
    [(0, 10, [spectrum.SpectrumDimension(i) for i in [31, 30, 30, 30, 30, 29, 28, 27, 28, 27]]),
     (1, 10, [spectrum.SpectrumDimension(i) for i in [27, 28, 27, 26, 25, 26, 27, 26, 25, 24]])]

)
def test_mcmc(seed, T, expected):
    target = spectrum.SpectrumDimensionTarget(10, 3.4)  # NB with mean 10 and variance 3.4^2
    initialProposal = spectrum.SpectrumDimensionInitialProposal(50)  # Uniform sampling from 0-50

    specMCMC = DiscreteVariableMCMC(spectrum.SpectrumDimension, target, initialProposal)
    samples = specMCMC.sample(T, seed=seed)

    assert samples == expected
