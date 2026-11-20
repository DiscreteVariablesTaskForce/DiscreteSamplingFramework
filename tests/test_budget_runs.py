"""
Runs given a budget of row reads must stop where the budget is reached, record
at the row-read marks, and be the same chains a run set by iterations gives:
the budget decides where a chain stops, never what it does before then.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                "examples", "incremental_decision_tree"))

import analyse_results as ar  # noqa: E402
import sampler_diagnostics as sd  # noqa: E402
from discretesampling.base.algorithms import DiscreteVariableMCMC  # noqa: E402
from discretesampling.domain import incremental_decision_tree as idt  # noqa: E402


def cfg(**overrides):
    base = dict(sd.DEFAULT_CFG, dataset="wine", sampler="mcmc", proposal="MH",
                seed=3, iters=100_000, steps=1_000, particles=8, ss_prop=0.25,
                min_data=5, lam=6.0, max_tree_size=20, record_moves=False)
    return dict(base, **overrides)


def test_a_callback_that_returns_true_stops_mcmc():
    from sklearn.datasets import load_wine
    X, y = load_wine(return_X_y=True)
    problem = idt.IncrementalTreeProblem(X, y, lam=6.0)
    target = idt.IncrementalTreeTarget(problem)
    mcmc = DiscreteVariableMCMC(idt.IncrementalTree, target,
                                idt.IncrementalTreeInitialProposal(problem, n_nodes=0))
    seen = []

    def callback(i, current, accepted):
        seen.append(i)
        return i == 9

    mcmc.sample(1_000, seed=1, verbose=False, callback=callback)
    assert seen == list(range(10))
    assert 0.0 <= mcmc.acceptance_rate <= 1.0


@pytest.mark.parametrize("proposal", ["MH", "DA", "HINTS"])
def test_mcmc_stops_at_the_budget_and_records_once_per_mark(proposal):
    budget, records = 400_000, 50
    knobs = dict(levels=2, branching=2) if proposal == "HINTS" else {}
    out = sd.run_mcmc(cfg(proposal=proposal, budget=budget, records=records, **knobs))
    reads, iters = out["row_reads"], out["iteration"]
    marks = sd.budget_marks(cfg(budget=budget, records=records))

    # One record per mark, each at or past its mark, the columns being those
    # of the records.
    assert len(reads) == len(iters) == len(out["state_iters"]) == records
    assert np.array_equal(iters, out["state_iters"])
    assert np.all(reads >= marks) and np.all(np.diff(iters) >= 0)
    # It stops at the iteration that reaches the budget, its last record.
    assert out["iterations"] == iters[-1] + 1
    assert 0 <= out["accepted_full"] <= out["full_evals"] <= out["iterations"]


def test_a_budget_run_is_the_same_chain_as_an_iteration_run():
    budget, records = 300_000, 40
    by_budget = sd.run_mcmc(cfg(budget=budget, records=records))
    n = int(by_budget["iterations"])
    by_iters = sd.run_mcmc(cfg(iters=n, store_every=1))
    marks = sd.budget_marks(cfg(budget=budget, records=records))
    full = by_iters["row_reads"]
    # The iteration-run reads the same rows, the budget run stopped at the
    # first iteration to reach the budget, and each record is the first
    # iteration to reach its mark.
    assert full[-1] >= budget and (n == 1 or full[-2] < budget)
    for it, reads, mark in zip(by_budget["iteration"], by_budget["row_reads"], marks):
        assert full[it] == reads >= mark
        assert it == 0 or full[it - 1] < mark
    # And it holds the same tree at the end.
    assert by_iters["n_nodes"][-1] == by_budget["n_nodes"][-1]


def test_smc_stops_at_the_budget_and_keeps_every_step():
    budget, records = 100_000, 10
    out = sd.run_smc(cfg(sampler="smc", budget=budget, records=records))
    steps = len(out["row_reads"])
    assert out["row_reads"][-1] >= budget and out["row_reads"][-2] < budget
    assert len(out["ess"]) == steps < 1_000
    assert out["state_iters"][-1] == steps - 1
    assert len(out["state_iters"]) == records


def test_budget_is_checked():
    with pytest.raises(ValueError):
        sd.validate_cfg(dict(sd.DEFAULT_CFG, budget=-1))
    with pytest.raises(ValueError):
        sd.validate_cfg(dict(sd.DEFAULT_CFG, records=0))


def test_the_analysis_reads_a_budget_run():
    out = sd.run_mcmc(cfg(budget=300_000, records=40))
    X_train = sd.load_data("wine")[0]
    lp_vals = -np.log(X_train.max(axis=0) - X_train.min(axis=0))
    summary = ar.summarise_mcmc_run(out, 6.0, lp_vals, -np.log(X_train.shape[1]))
    assert summary["iters"] == out["iterations"]
    assert np.array_equal(summary["rec_reads"], out["row_reads"])
    assert summary["total_reads"] == out["row_reads"][-1]
    assert summary["full_evals"] == out["full_evals"]


def test_the_analysis_reads_an_smc_budget_run():
    out = sd.run_smc(cfg(sampler="smc", budget=100_000, records=10))
    X_train = sd.load_data("wine")[0]
    lp_vals = -np.log(X_train.max(axis=0) - X_train.min(axis=0))
    summary = ar.summarise_smc_run(out, 6.0, lp_vals, -np.log(X_train.shape[1]))
    assert summary["budget"] == 100_000
    assert len(summary["rec_wlp"]) == 10
    assert summary["total_reads"] == out["row_reads"][-1] >= 100_000


def test_the_report_takes_runs_of_different_lengths(capsys):
    runs = [sd.run_smc(cfg(sampler="smc", budget=100_000, records=5, seed=s)) for s in (1, 2)]
    assert len(runs[0]["ess_history"]) != len(runs[1]["ess_history"])
    sd.report(cfg(sampler="smc"), runs)
    assert "ESS" in capsys.readouterr().out
