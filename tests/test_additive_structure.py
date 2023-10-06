import pytest
import numpy as np
import pandas as pd
from discretesampling.base.random import RNG
from discretesampling.domain.additive_structure import (
    AdditiveStructure, AdditiveStructureProposal, AdditiveStructureInitialProposal,
    AdditiveStructureTarget
)


def func(x):
    return (x.iloc[:, 0] + x.iloc[:, 1]) * (x.iloc[:, 2] + x.iloc[:, 4]) * x.iloc[:, 3]


def getData(seed):
    n = 1000
    nprng = np.random.default_rng(seed)
    x = nprng.uniform(-3, 3, n)
    x.shape = (200, 5)
    x_train = pd.DataFrame(x)
    y_train = func(x_train)
    data = list([x_train, y_train])  # some data defining the target
    return data


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
    prop = AdditiveStructureProposal()
    logprob = prop.eval(ad_start, ad_end)
    np.testing.assert_almost_equal(logprob, expected)


@pytest.mark.parametrize(
    "seed,ad_start,expected",
    [(0, AdditiveStructure([[1, 2, 3, 4, 5]]), AdditiveStructure([[1, 2, 3], [4, 5]])),
     (1, AdditiveStructure([[1, 2], [3, 4, 5]]), AdditiveStructure([[1, 2], [3], [4, 5]])),
     (2, AdditiveStructure([[1, 2], [3], [4, 5]]), AdditiveStructure([[1, 2, 3], [4, 5]])),
     (3, AdditiveStructure([[1], [2], [3], [4], [5]]), AdditiveStructure([[1], [2], [3, 5], [4]]))]
)
def test_additive_structure_proposal_sample(seed, ad_start, expected):
    prop = AdditiveStructureProposal()
    sampled = prop.sample(ad_start, rng=RNG(seed))
    assert sampled == expected


@pytest.mark.parametrize(
    "elems,ad,expected",
    [([1, 2, 3, 4, 5], AdditiveStructure([[1, 2, 3], [4, 5]]), -np.log(52))]
)
def test_additive_structure_initial_proposal_eval(elems, ad, expected):
    prop = AdditiveStructureInitialProposal(elems)
    logprob = prop.eval(ad)
    np.testing.assert_almost_equal(logprob, expected)


@pytest.mark.parametrize(
    "seed,elems,expected",
    [(0, [1, 2, 3, 4, 5], AdditiveStructure([[1], [2, 3], [4, 5]])),
     (1, [1, 2, 3, 4, 5], AdditiveStructure([[1, 2], [3], [4], [5]])),
     (2, [1, 2, 3, 4, 5], AdditiveStructure([[1, 2, 3, 5], [4]])),
     (3, [1, 2, 3, 4, 5], AdditiveStructure([[1, 2, 3, 4, 5]]))]
)
def test_additive_structure_initial_proposal_sample(seed, elems, expected):
    prop = AdditiveStructureInitialProposal(elems)
    sampled = prop.sample(rng=RNG(seed))
    assert sampled == expected


@pytest.mark.parametrize(
    "seed,x,expected",
    [(0, AdditiveStructure([[1], [2, 3], [4, 5]]), 1/52),
     (1, AdditiveStructure([[1, 2], [3], [4], [5, 6]]), 1/203),
     (2, AdditiveStructure([[1, 2, 3, 5], [4], [6, 7]]), 1/877),
     (3, AdditiveStructure([[1, 2, 3]]), 1/5)]
)
def test_additive_structure_target_evaluatePrior(seed, x, expected):
    data = getData(seed)
    target = AdditiveStructureTarget(data)
    logprob = target.evaluatePrior(x)
    assert logprob == expected


@pytest.mark.parametrize(
    "seed,y,mean,var,expected",
    [(0, 0.0, 0.0, 1.0, -0.9189385332046727),
     (1, -3.0, 0.0, 1.0, -5.418938533204672),
     (2, -3.0, 5.0, 10.1, -3.5451686928621013)]
)
def test_additive_structure_target_log_likelihood(seed, y, mean, var, expected):
    data = getData(seed)
    target = AdditiveStructureTarget(data)
    logprob = target.log_likelihood(y, mean, var)
    np.testing.assert_almost_equal(logprob, expected)


@pytest.mark.parametrize(
    "seed,structure,expected",
    [(0, [[1, 2, 3, 4, 5]], 0.4172766628123803),
     (1, [[1, 2, 3, 4, 5]], 2.3551424667500584),
     (2, [[1, 2], [3], [4, 5]], 0.0)]
)
def test_additive_structure_target_evaluate(seed, structure, expected):
    data = getData(seed)
    target = AdditiveStructureTarget(data)
    xtrain = data[0]
    calc = target.evaluate(structure, xtrain[0])
    np.testing.assert_almost_equal(calc, expected)


@pytest.mark.parametrize(
    "seed,x,expected",
    [(0, AdditiveStructure([[1], [2, 3], [4, 5]]), -165195.03483191656),
     (1, AdditiveStructure([[1, 2], [3], [4], [5]]), -202235.73413464168),
     (2, AdditiveStructure([[1, 2, 3, 5], [4]]), -177047.48120419323),
     (3, AdditiveStructure([[1, 2, 3]]), -187870.65855094)]
)
def test_additive_structure_target_eval(seed, x, expected):
    data = getData(seed)
    target = AdditiveStructureTarget(data)
    logprob = target.eval(x)
    np.testing.assert_almost_equal(logprob, expected)


@pytest.mark.parametrize(
    "ad",
    [(AdditiveStructure([[1, 2], [3, 4, 5]])),
     (AdditiveStructure([[1, 2, 3], [4, 5]]))]
)
def test_encode_decode_additivestructure(ad):
    encoded_ad = ad.encode(ad)
    decoded_ad = ad.decode(encoded_ad, ad)
    assert ad == decoded_ad
