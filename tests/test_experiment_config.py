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


@pytest.mark.parametrize("bad", [
    dict(init_tree_size=-1),
    dict(init_tree_size=2.5),
    dict(init_tree_size=11, max_tree_size=10),
])
def test_starting_tree_sizes_that_cannot_be_run_are_refused(bad):
    with pytest.raises(ValueError):
        sd.validate_cfg(cfg(**bad))


def test_the_starting_tree_size_is_in_the_name_unless_it_is_a_stump():
    import results_io as rio
    assert rio.experiment_filename("susy", "mcmc", "DA", 0.25) == "susy_mcmc_DA_25.h5"
    assert (rio.experiment_filename("susy", "mcmc", "DA", 0.25, 10)
            == "susy_mcmc_DA_25_init10.h5")
    for key in [("susy", 0.125, 0), ("susy", 0.125, 10), ("my_data", 1.0, 3)]:
        assert rio.parse_experiment(rio.experiment_name(*key)) == key
    assert rio.parse_experiment_name("susy_12p5_init10") == ("susy", 0.125)


def test_runs_from_different_starting_sizes_write_different_files():
    from run_experiments import check_for_collisions, experiment_files
    a = dict(samplers=["mcmc"], proposals=["MH"])
    b = dict(a, init_tree_size=5)
    assert experiment_files(a) != experiment_files(b)
    check_for_collisions([a, b])


TREE = [[0, 1, 2, 0, 13.0], [2, 5, 6, 6, 2.0]]


@pytest.mark.parametrize("bad", [
    dict(init_tree=TREE, init_tree_size=2),              # two starts
    dict(init_tree_name="mine"),                         # a name for no tree
    dict(init_tree=TREE, init_tree_name="my_tree"),      # '_' splits the names
    dict(init_tree=TREE, max_tree_size=1),
    dict(init_tree=[]),
    dict(init_tree=[[1, 2, 3, 0, 13.0]]),                # no root
])
def test_given_starting_trees_that_cannot_be_run_are_refused(bad):
    with pytest.raises(ValueError):
        sd.validate_cfg(cfg(**bad))


def test_a_given_starting_tree_is_in_the_name():
    import results_io as rio
    named = cfg(init_tree=TREE, init_tree_name="mine", proposals=["DA"], ss_prop=0.25)
    sd.validate_cfg(named)
    key = rio.experiment_key(named)
    assert key == ("wine", 0.25, "mine")
    assert rio.experiment_name(*key) == "wine_25_init-mine"
    assert rio.parse_experiment("wine_25_init-mine") == key
    assert (rio.experiment_filename("wine", "mcmc", "DA", 0.25, "mine")
            == "wine_mcmc_DA_25_init-mine.h5")
    # A name that is all digits is still a name, not a size.
    assert rio.parse_experiment("wine_25_init-10") == ("wine", 0.25, "10")
    assert rio.parse_experiment("wine_25_init10") == ("wine", 0.25, 10)


def test_an_unnamed_starting_tree_is_named_after_its_rows():
    import results_io as rio
    from run_experiments import check_for_collisions
    a = dict(samplers=["mcmc"], proposals=["MH"], init_tree=TREE)
    same = dict(a, init_tree=[row + [0] for row in TREE[::-1]])   # reordered, with depths
    other = dict(a, init_tree=[[0, 1, 2, 0, 13.5]])
    name = rio.init_of(cfg(**a))
    assert rio.INIT_NAME.fullmatch(name)
    assert rio.init_of(cfg(**same)) == name
    assert rio.init_of(cfg(**other)) != name
    check_for_collisions([a, other, dict(a, init_tree=None)])
    with pytest.raises(SystemExit):
        check_for_collisions([a, same])
    # What config.json hands back names the tree the same way.
    import json
    assert rio.init_of(json.loads(json.dumps(cfg(**a)))) == name


def test_the_harness_starts_from_the_tree_it_is_given():
    X, y = load_wine(return_X_y=True)
    problem, _, _ = sd.build(cfg(proposal="MH"), X, y)
    start = sd.initial_proposal(cfg(init_tree=TREE), problem).sample()
    assert [row[:5] for row in start.tree] == TREE
    assert len(sd.initial_proposal(cfg(init_tree_size=3), problem).sample().tree) == 3
    assert len(sd.initial_proposal(cfg(), problem).sample().tree) == 0



def test_a_stored_tree_can_be_handed_back_as_the_start(tmp_path):
    import numpy as np
    import results_io as rio
    chain = {k: v for k, v in cfg(dataset="wine", steps=6, particles=6,
                                  init_tree_size=2).items()
             if k not in sd._SWEEP_KEYS}
    out = sd.run(dict(chain, sampler="smc", proposal="MH", seed=0))
    path = str(tmp_path / "wine_smc_MH_100.h5")
    rio.save_experiment_hdf5(path, [out])

    series = rio.states_of(out)
    heaviest = int(np.argmax(series.weights(5)))
    want = [[int(v) for v in row[:4]] + [float(row[4])]
            for row in series.state(series.slots(5)[heaviest]).tree]
    rows = rio.stored_tree(path, run=0, record=5)
    assert rows == want
    assert rio.stored_tree(path, record=5, particle=0) == [
        [int(v) for v in row[:4]] + [float(row[4])]
        for row in series.state(series.slots(5)[0]).tree]

    # And it is a start the harness takes: same tree, same leaves.
    sd.validate_cfg(cfg(init_tree=rows))
    X_train, _, y_train, _ = sd.load_data("wine")
    problem, _, _ = sd.build(cfg(proposal="MH"), X_train, y_train)
    start = sd.initial_proposal(cfg(init_tree=rows), problem).sample()
    assert sorted(row[:5] for row in start.tree) == sorted(rows)
