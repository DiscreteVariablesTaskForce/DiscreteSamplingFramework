import pytest
import numpy as np
from discretesampling.base.random import RNG
from discretesampling.domain.additive_structure import (
    AdditiveStructure, AdditiveStructureProposal, AdditiveStructureInitialProposal,
    AdditiveStructureTarget
)


@pytest.mark.parametrize(
    "ad,expected",
    [(AdditiveStructure([[1, 2, 3, 4, 5]]), 1),
     (AdditiveStructure([[1, 2], [3, 4, 5]]), 2),
     (AdditiveStructure([[1, 2, 3], [4, 5]]), 2),
     (AdditiveStructure([[1], [2], [3], [4], [5]]), 5)]
)
def test_additive_structure_proposal_norm(ad, expected):
    x = AdditiveStructureProposal.norm(ad)
    assert x == expected


@pytest.mark.parametrize(
    "ad_a,ad_b,expected",
    [(AdditiveStructure([[1, 2, 3, 4, 5]]), AdditiveStructure([[1, 2, 3], [4, 5]]), True),
     (AdditiveStructure([[1, 2], [3, 4, 5]]), AdditiveStructure([[1, 2, 3], [4, 5]]), False),
     (AdditiveStructure([[1, 2], [3], [4, 5]]), AdditiveStructure([[1, 2], [3], [4, 5]]), False),
     (AdditiveStructure([[1], [2], [3], [4], [5]]), AdditiveStructure([[1], [2], [3], [4, 5]]), True)]
)
def test_additive_structure_proposal_heuristic(ad_a, ad_b, expected):
    a = AdditiveStructureProposal.norm(ad_a)
    b = AdditiveStructureProposal.norm(ad_b)
    x = AdditiveStructureProposal.heuristic(a, b)
    y = AdditiveStructureProposal.heuristic(b, a)
    assert x == expected and y == expected


@pytest.mark.parametrize(
    "ad_start,ad_end,expected",
    [(AdditiveStructure([[1, 2, 3, 4, 5]]), AdditiveStructure([[1, 2, 3], [4, 5]]), np.log(1/15)),
     (AdditiveStructure([[1, 2], [3, 4, 5]]), AdditiveStructure([[1, 2, 3], [4, 5]]), -np.inf),
     (AdditiveStructure([[1, 2], [3], [4, 5]]), AdditiveStructure([[1, 2], [3], [4, 5]]), -np.inf),
     (AdditiveStructure([[1], [2], [3], [4], [5]]), AdditiveStructure([[1], [2], [3], [4, 5]]), np.log(0.1))]
)
def test_additive_structure_proposal_eval(ad_start, ad_end, expected):
    prop = AdditiveStructureProposal(ad_start)
    logprob = prop.eval(ad_end)
    np.testing.assert_almost_equal(logprob, expected)


@pytest.mark.parametrize(
    "seed,ad_start,expected",
    [(0, AdditiveStructure([[1, 2, 3, 4, 5]]), AdditiveStructure([[1, 2, 3], [4, 5]])),
     (1, AdditiveStructure([[1, 2], [3, 4, 5]]), AdditiveStructure([[1, 2], [3], [4, 5]])),
     (2, AdditiveStructure([[1, 2], [3], [4, 5]]), AdditiveStructure([[1, 2, 3], [4, 5]])),
     (3, AdditiveStructure([[1], [2], [3], [4], [5]]), AdditiveStructure([[1], [2], [3, 5], [4]]))]
)
def test_additive_structure_proposal_sample(seed, ad_start, expected):
    prop = AdditiveStructureProposal(ad_start, rng=RNG(seed))
    sampled = prop.sample()
    assert sampled == expected


@pytest.mark.parametrize(
    "ad",
    [(AdditiveStructure([[1, 2], [3, 4, 5]])),
     (AdditiveStructure([[1, 2, 3], [4, 5]]))]
)
def test_encode_decode_additivestructure(ad):
    encoded_ad = ad.encode(ad)
    decoded_ad = ad.decode(encoded_ad, ad)
    assert ad == decoded_ad
