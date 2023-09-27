import pytest
import numpy as np
from discretesampling.base.algorithms import DiscreteVariableSMC
from discretesampling.base.util import gather_all
from discretesampling.base.executor.executor_MPI import Executor_MPI
from discretesampling.domain import spectrum
from discretesampling.base.output import SMCOutput

# Test reproducibility


@pytest.mark.parametrize(
    "seed,T,N,expected",
    [
        (0, 3, 16, SMCOutput(
            [spectrum.SpectrumDimension(i) for i in [15, 13, 15, 6, 6, 14, 8, 8, 8, 6, 6, 6, 14, 14, 12, 10]],
            np.array([-3.3423949417970995, -2.6927167512983843, -3.3423949417970995, -2.5485352600349263, -2.5485352600349263,
                      -1.7791312350561306, -2.807685770214025, -2.807685770214025, -2.807685770214025, -3.2906867661536747,
                      -3.2906867661536747, -3.2906867661536747, -3.286288420776657, -3.286288420776657, -2.753770144750778,
                      -2.4895429738738315])
        )),
        (1, 3, 16, SMCOutput(
            [spectrum.SpectrumDimension(i) for i in [13, 15, 15, 6, 2, 6, 18, 8, 6, 6, 6, 8, 14, 12, 10, 12]],
            np.array([-2.577088421206954, -3.2267666117056693, -3.2267666117056693, -2.432906929943496, -5.389379226012296,
                      -2.432906929943496, -3.37139647864896, -2.692057440122595, -3.1750584360622445, -3.1750584360622445,
                      -3.1750584360622445, -2.692057440122595, -3.1706600906852267, -2.6381418146593476, -2.3739146437824012,
                      -2.6381418146593476])
        )
        )]
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
    [
        (0, 3, 16, SMCOutput(
            [spectrum.SpectrumDimension(i) for i in [15, 13, 15, 6, 6, 14, 8, 8, 8, 6, 6, 6, 14, 14, 12, 10]],
            np.array([-3.3423949417970995, -2.6927167512983843, -3.3423949417970995, -2.5485352600349263, -2.5485352600349263,
                      -1.7791312350561306, -2.807685770214025, -2.807685770214025, -2.807685770214025, -3.2906867661536747,
                      -3.2906867661536747, -3.2906867661536747, -3.286288420776657, -3.286288420776657, -2.753770144750778,
                      -2.4895429738738315])
        )),
        (1, 3, 16, SMCOutput(
            [spectrum.SpectrumDimension(i) for i in [13, 15, 15, 6, 2, 6, 18, 8, 6, 6, 6, 8, 14, 12, 10, 12]],
            np.array([-2.577088421206954, -3.2267666117056693, -3.2267666117056693, -2.432906929943496, -5.389379226012296,
                      -2.432906929943496, -3.37139647864896, -2.692057440122595, -3.1750584360622445, -3.1750584360622445,
                      -3.1750584360622445, -2.692057440122595, -3.1706600906852267, -2.6381418146593476, -2.3739146437824012,
                      -2.6381418146593476])
        ))
    ]
)
def test_smc_MPI(seed, T, N, expected):
    target = spectrum.SpectrumDimensionTarget(10, 3.4)  # NB with mean 10 and variance 3.4^2
    initialProposal = spectrum.SpectrumDimensionInitialProposal(50)  # Uniform sampling from 0-50

    exec = Executor_MPI()
    specSMC = DiscreteVariableSMC(spectrum.SpectrumDimension, target, initialProposal, exec=exec)
    samples = specSMC.sample(T, N, seed=seed)
    assert samples == expected
