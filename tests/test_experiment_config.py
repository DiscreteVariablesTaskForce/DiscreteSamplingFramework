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
    from run_experiments import DEFAULT_FILE, check_for_collisions, load_experiments
    here = os.path.dirname(DEFAULT_FILE)
    for path in (DEFAULT_FILE, os.path.join(here, "experiments_reference.py"),
                 os.path.join(here, "experiments_hints_shapes.py"),
                 *(os.path.join(here, f"experiments_budget_hierarchies{suffix}.py")
                   for suffix in ("", "_one_level", "_branch4", "_branch8"))):
        experiments = load_experiments(path)
        for entry in experiments:
            sd.validate_cfg(cfg(**entry))
        check_for_collisions(experiments)


@pytest.mark.parametrize("bad", [
    dict(hierarchy=[1, 2, 4], levels=2),
    dict(hierarchy=[1, 2, 4], ss_prop=0.25),
    dict(hierarchy=[1, 1, 8], ss_prop=1 / 8),
    dict(hierarchy="1,2,4", ss_prop=1 / 8),
])
def test_a_hierarchy_that_cannot_be_run_as_given_is_refused(bad):
    with pytest.raises(ValueError):
        sd.validate_cfg(cfg(**dict(dict(proposals=["HINTS"]), **bad)))


def test_a_hierarchy_with_its_own_ss_prop_is_accepted():
    sd.validate_cfg(cfg(proposals=["HINTS"], hierarchy=[2, 4, 8], ss_prop=1 / 64))


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
    assert len(sd.initial_proposal(cfg(init_tree_size=0), problem).sample().tree) == 0


def test_by_default_the_harness_draws_every_start_from_the_prior():
    from discretesampling.base.random import RNG
    X, y = load_wine(return_X_y=True)
    problem, _, _ = sd.build(cfg(proposal="MH"), X, y)
    init = sd.initial_proposal(cfg(), problem)
    assert init.from_prior
    starts = [init.sample(RNG(seed)) for seed in range(10)]
    assert len({tuple(map(tuple, x.tree)) for x in starts}) > 1


def test_one_task_of_an_array_records_the_whole_list(tmp_path):
    """A SLURM array runs one --index per task into one run directory: each
    task must run only its own experiment, but config.json must still list
    them all, or the evaluation afterwards would only know about one."""
    import json
    import subprocess
    script = os.path.join(os.path.dirname(__file__), os.pardir, "examples",
                          "incremental_decision_tree", "run_experiments.py")
    experiments = tmp_path / "experiments.py"
    experiments.write_text(
        "EXPERIMENTS = [dict(dataset='wine', samplers=['mcmc'], proposals=[p], "
        "ss_prop=0.25, chains=1, iters=5, store_every=1) for p in ('MH', 'DA')]\n")

    def run(*args):
        return subprocess.run(["python", script, "--file", str(experiments), *args],
                              capture_output=True, text=True, check=True).stdout

    assert run("--count").strip() == "2"
    run("--index", "1", "--jobs", "1", "--run-id", "arr", "--results-root", str(tmp_path))
    out = tmp_path / "arr"
    assert len(json.loads((out / "config.json").read_text())["experiments"]) == 2
    assert sorted(p.name for p in out.glob("*.h5")) == ["wine_mcmc_DA_25_init-prior.h5"]
    # --prepare splits each dataset of the list once, into the run's directory.
    run("--prepare", "--run-id", "arr", "--results-root", str(tmp_path))
    assert sorted(p.name for p in (out / "data").iterdir()) == ["wine.npz"]


def test_prepared_data_is_the_split_itself(tmp_path, monkeypatch):
    """The jobs read a split prepared at submission; it has to be the split
    load_data computes, and a missing one has to fail rather than quietly
    re-reading the CSV on every compute node."""
    import numpy as np
    sd.prepare_data("wine", str(tmp_path))
    monkeypatch.setenv(sd.DATA_CACHE_ENV, str(tmp_path))
    for got, want in zip(sd.load_data("wine"), sd.split_data("wine")):
        assert np.array_equal(got, want)
    with pytest.raises(FileNotFoundError):
        sd.load_data("digits")


@pytest.mark.parametrize("sampler", ["mcmc", "smc"])
def test_runs_record_their_row_reads(sampler):
    import numpy as np
    chain = {k: v for k, v in cfg(dataset="wine", iters=20, steps=4, particles=4,
                                  record_states=False, record_moves=False).items()
             if k not in sd._SWEEP_KEYS}
    out = sd.run(dict(chain, sampler=sampler, proposal="HINTS", seed=0))
    reads = np.asarray(out["row_reads"])
    assert len(reads) == (20 if sampler == "mcmc" else 4)
    assert reads[-1] > 0
    assert np.all(np.diff(reads) >= 0)


def test_starts_drawn_from_the_prior_are_in_the_name():
    import results_io as rio
    key = rio.experiment_key(sd.validate_cfg(cfg(proposals=["DA"], ss_prop=0.25)))
    assert key == ("wine", 0.25, rio.PRIOR_INIT)
    assert rio.experiment_name(*key) == "wine_25_init-prior"
    assert rio.parse_experiment("wine_25_init-prior") == key
    assert (rio.experiment_filename("wine", "smc", "DA", 0.25, rio.PRIOR_INIT)
            == "wine_smc_DA_25_init-prior.h5")
    # A given tree may not take the prior's tag.
    with pytest.raises(ValueError):
        sd.validate_cfg(cfg(init_tree=TREE, init_tree_name=rio.PRIOR_INIT))


def test_stored_configs_keep_meaning_what_they_meant():
    """A config.json entry with no init_tree_size was written when leaving it
    out meant a stump, and still names the stump's files; entries written now
    carry it, so they name the prior's."""
    import results_io as rio
    from run_experiments import stored_entries
    entry = dict(dataset="wine", samplers=["mcmc"], proposals=["DA"], ss_prop=0.25)
    assert rio.experiment_names({"experiments": [entry]}, sd.DEFAULT_CFG) == ["wine_25"]
    assert rio.init_of(entry) == 0
    stored = stored_entries([entry])
    assert stored[0]["init_tree_size"] is None
    assert rio.experiment_names({"experiments": stored}, sd.DEFAULT_CFG) == ["wine_25_init-prior"]
    assert rio.init_of(stored[0]) == rio.PRIOR_INIT
    sized = stored_entries([dict(entry, init_tree_size=5)])
    assert rio.experiment_names({"experiments": sized}, sd.DEFAULT_CFG) == ["wine_25_init5"]



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
