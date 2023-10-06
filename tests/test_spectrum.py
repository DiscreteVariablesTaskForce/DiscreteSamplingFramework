import pytest
import numpy as np
from discretesampling.base.random import RNG
from discretesampling.domain.spectrum import (
    SpectrumDimension, SpectrumDimensionInitialProposal, SpectrumDimensionProposal, SpectrumDimensionTarget
)


@pytest.mark.parametrize(
    "spectrum_a,spectrum_b,expected",
    [(SpectrumDimension(1), SpectrumDimension(2), False),
     (SpectrumDimension(1), SpectrumDimension(1), True),
     (SpectrumDimension(1), SpectrumDimension(1.0), True)]
)
def test_spectrum_eq(spectrum_a, spectrum_b, expected):
    result = spectrum_a == spectrum_b
    assert result == expected


@pytest.mark.parametrize(
    "spectrum",
    [(SpectrumDimension(1)),
     (SpectrumDimension(10))]
)
def test_encode_decode_spectrum(spectrum):
    encoded_spectrum = spectrum.encode(spectrum)
    decoded_spectrum = spectrum.decode(encoded_spectrum, spectrum)
    assert spectrum == decoded_spectrum


@pytest.mark.parametrize(
    "spectrum,expected",
    [(SpectrumDimension(1), 1),
     (SpectrumDimension(2), 2)]
)
def test_spectrum_proposal_norm(spectrum, expected):
    assert SpectrumDimensionProposal.norm(spectrum) == expected


@pytest.mark.parametrize(
    "a,b,expected",
    [(1, 2, True),
     (3, 2, True),
     (1, 1, False),
     (2, 5, False)]
)
def test_spectrum_proposal_heuristic(a, b, expected):
    assert SpectrumDimensionProposal.heuristic(a, b) == expected


@pytest.mark.parametrize(
    "start_spectrum,seed,expected",
    [(SpectrumDimension(1), 1, SpectrumDimension(2)),
     (SpectrumDimension(1), 2, SpectrumDimension(2)),
     (SpectrumDimension(2), 3, SpectrumDimension(1)),
     (SpectrumDimension(2), 4, SpectrumDimension(3))]
)
def test_spectrum_proposal_sample(start_spectrum, seed, expected):
    prop = SpectrumDimensionProposal()
    sampled = prop.sample(start_spectrum, rng=RNG(seed))
    assert sampled == expected


@pytest.mark.parametrize(
    "start_spectrum,end_spectrum,expected",
    [(SpectrumDimension(1), SpectrumDimension(2), np.log(1.0)),
     (SpectrumDimension(2), SpectrumDimension(1), np.log(0.5)),
     (SpectrumDimension(2), SpectrumDimension(3), np.log(0.5)),
     (SpectrumDimension(3), SpectrumDimension(5), -np.inf),
     (SpectrumDimension(3), SpectrumDimension(1), -np.inf),
     (SpectrumDimension(1), SpectrumDimension(0), -np.inf)]
)
def test_spectrum_proposal_eval(start_spectrum, end_spectrum, expected):
    prop = SpectrumDimensionProposal()
    logprob = prop.eval(start_spectrum, end_spectrum)
    assert logprob == expected


@pytest.mark.parametrize(
    "max,seed,expected",
    [(10, 1, SpectrumDimension(6)),
     (1, 2, SpectrumDimension(1)),
     (100, 3, SpectrumDimension(9)),
     (100, 4, SpectrumDimension(95)),
     (100, 5, SpectrumDimension(81))]
)
def test_spectrum_initial_proposal_sample(max, seed, expected):
    prop = SpectrumDimensionInitialProposal(max)
    sampled = prop.sample(rng=RNG(seed))
    assert sampled == expected


@pytest.mark.parametrize(
    "max,sampled,expected",
    [(10, SpectrumDimension(6), -np.log(10)),
     (1, SpectrumDimension(1), -np.log(1)),
     (100, SpectrumDimension(9), -np.log(100)),
     (100, SpectrumDimension(95), -np.log(100)),
     (100, SpectrumDimension(81), -np.log(100))]
)
def test_spectrum_initial_proposal_eval(max, sampled, expected):
    prop = SpectrumDimensionInitialProposal(max)
    logprob = prop.eval(sampled)
    np.testing.assert_allclose(logprob, expected)


@pytest.mark.parametrize(
    "mu,sigma,x,expected",
    [(10, 3.4, SpectrumDimension(6), -2.6986342482951375),
     (10, 3.4, SpectrumDimension(1), -7.1350582573962305),
     (10, 3.4, SpectrumDimension(15), -3.30085775972983),
     (50, 8, SpectrumDimension(45), -3.140272778089198),
     (100, 11, SpectrumDimension(42), -21.374638449414263)]
)
def test_spectrum_target_eval(mu, sigma, x, expected):
    target = SpectrumDimensionTarget(mu, sigma)
    logprob = target.eval(x)
    np.testing.assert_allclose(logprob, expected)
