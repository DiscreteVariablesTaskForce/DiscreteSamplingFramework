import pytest
from discretesampling.base.algorithms import DiscreteVariableSMC
from discretesampling.domain import spectrum

# Test reproducibility


@pytest.mark.parametrize(
    "seed,T,N,expected",
    [(0, 3, 10, [spectrum.SpectrumDimension(i) for i in [6, 6, 8, 8, 12, 10, 12, 12, 14, 14]]),
     (1, 3, 10, [spectrum.SpectrumDimension(i) for i in [12, 14, 14, 12, 10, 12, 19, 14, 14, 16]])]
)
def test_smc(seed, T, N, expected):
    target = spectrum.SpectrumDimensionTarget(10, 3.4)  # NB with mean 10 and variance 3.4^2
    initialProposal = spectrum.SpectrumDimensionInitialProposal(50)  # Uniform sampling from 0-50

    specSMC = DiscreteVariableSMC(spectrum.SpectrumDimension, target, initialProposal)
    samples = specSMC.sample(T, N, seed=seed)

    assert samples == expected
