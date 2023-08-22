import pytest
from discretesampling.base.algorithms import DiscreteVariableSMC
from discretesampling.base.util import gather_all
from discretesampling.base.executor.executor_MPI import Executor_MPI
from discretesampling.domain import spectrum

# Test reproducibility


@pytest.mark.parametrize(
    "seed,T,N,expected",
    [(0, 3, 16, [spectrum.SpectrumDimension(i) for i in [15, 13, 15, 6, 6, 14, 8, 8, 8, 6, 6, 6, 14, 14, 12, 10]]),
     (1, 3, 16, [spectrum.SpectrumDimension(i) for i in [13, 15, 15, 6, 2, 6, 18, 8, 6, 6, 6, 8, 14, 12, 10, 12]])]
)
def test_smc(seed, T, N, expected):
    target = spectrum.SpectrumDimensionTarget(10, 3.4)  # NB with mean 10 and variance 3.4^2
    initialProposal = spectrum.SpectrumDimensionInitialProposal(50)  # Uniform sampling from 0-50

    specSMC = DiscreteVariableSMC(spectrum.SpectrumDimension, target, initialProposal)
    samples = specSMC.sample(T, N, seed=seed)

    assert samples == expected


@pytest.mark.mpi
@pytest.mark.parametrize(
    "seed,T,N,expected",
    [(0, 3, 16, [spectrum.SpectrumDimension(i) for i in [15, 13, 15, 6, 6, 14, 8, 8, 8, 6, 6, 6, 14, 14, 12, 10]]),
     (1, 3, 16, [spectrum.SpectrumDimension(i) for i in [13, 15, 15, 6, 2, 6, 18, 8, 6, 6, 6, 8, 14, 12, 10, 12]])]
)
def test_smc_MPI(seed, T, N, expected):
    target = spectrum.SpectrumDimensionTarget(10, 3.4)  # NB with mean 10 and variance 3.4^2
    initialProposal = spectrum.SpectrumDimensionInitialProposal(50)  # Uniform sampling from 0-50

    exec = Executor_MPI()
    specSMC = DiscreteVariableSMC(spectrum.SpectrumDimension, target, initialProposal, exec=exec)
    local_samples = specSMC.sample(T, N, seed=seed)
    samples = gather_all(local_samples, exec)
    assert samples == expected
