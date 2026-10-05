"""
The experiment harness must run exactly what an experiment names, or refuse.

Every failure guarded here was silent before, and each one produced results
labelled with a sampler they were not sampled with.
"""
import os
import sys

import pytest
from sklearn.datasets import load_wine

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                "examples", "incremental_decision_tree"))

import sampler_diagnostics as sd  # noqa: E402
from discretesampling.domain import incremental_decision_tree as idt  # noqa: E402


def cfg(**overrides):
    return dict(sd.DEFAULT_CFG, **overrides)


def test_a_bare_string_names_one_proposal_not_its_letters():
    checked = sd.validate_cfg(cfg(proposals="HINTS", samplers="mcmc"))
    assert checked["proposals"] == ["HINTS"]
    assert checked["samplers"] == ["mcmc"]


@pytest.mark.parametrize("bad", [
    dict(proposals=["HINTS9"]),
    dict(proposals=[]),
    dict(samplers=["pmcmc"]),
    dict(levles=3, proposals=["HINTS"]),       # misspelt knob
    dict(levels=3, proposals=["MH", "DA"]),     # knob with no HINTS to take it
    dict(starve_penalty=0, proposals=["HINTS"]),   # knob of a removed proposal
])
def test_experiments_that_would_run_something_else_are_refused(bad):
    with pytest.raises(ValueError):
        sd.validate_cfg(cfg(**bad))


@pytest.mark.parametrize("name, cls", [
    ("MH", idt.IncrementalTreeProposal),
    ("DA", idt.DAProposal),
    ("FlatHINTS", idt.FlatHINTSProposal),
    ("HINTS", idt.HINTSProposal),
])
def test_build_makes_the_proposal_it_is_asked_for(name, cls):
    X, y = load_wine(return_X_y=True)
    _, _, proposal = sd.build(cfg(proposal=name), X, y)
    assert type(proposal) is cls


def test_build_has_no_fallback_proposal():
    X, y = load_wine(return_X_y=True)
    with pytest.raises(ValueError):
        sd.build(cfg(proposal="H"), X, y)


def test_the_shipped_experiment_list_is_valid():
    from run_experiments import DEFAULT_FILE, load_experiments
    for entry in load_experiments(DEFAULT_FILE):
        sd.validate_cfg(cfg(**entry))
