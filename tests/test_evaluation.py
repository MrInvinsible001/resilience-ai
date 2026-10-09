from __future__ import annotations

from types import SimpleNamespace
import subprocess
import sys

import pandas as pd

from resilience_ai.contracts import (
    ProcurementDecision,
    ScenarioMode,
)
from resilience_ai.evaluation import (
    AGGREGATE_COLUMNS,
    SUMMARY_COLUMNS,
    TRAJECTORY_COLUMNS,
    _recovery_time,
    _same_day_served,
    run_benchmark,
)
from resilience_ai.simulator import (
    SimulationConfig,
    generate_daily_demands,
)
from scripts.generate_dataset import generate


class NoOrderStrategy:
    def __init__(self):
        self.observations = []

    def propose(self, observation):
        self.observations.append(observation)
        return ProcurementDecision(day=observation.day, orders=())


def _result_with_backlogs(backlogs: list[int]):
    config = SimulationConfig(horizon_days=len(backlogs))
    steps = tuple(
        SimpleNamespace(day=day, backlog_end=backlog)
        for day, backlog in enumerate(backlogs, start=1)
    )
    return SimpleNamespace(config=config, steps=steps)


def test_generate_daily_demands_matches_dataset_seed_42(tmp_path):
    config = SimulationConfig(seed=42)
    generated = generate_daily_demands(config)
    generate(seed=42, output_dir=tmp_path)
    demand = pd.read_csv(tmp_path / "demand.csv")
    expected = demand.groupby("day")["demand"].sum().to_dict()
    assert generated == expected


def test_generate_daily_demands_is_reproducible():
    config = SimulationConfig(seed=123)
    assert generate_daily_demands(config) == generate_daily_demands(config)


def test_same_day_served_excludes_old_backlog():
    step = SimpleNamespace(demand=10, fulfilled=15, backlog_start=20)
    assert _same_day_served(step) == 0
    step = SimpleNamespace(demand=10, fulfilled=25, backlog_start=20)
    assert _same_day_served(step) == 5


def test_recovery_time_conventions():
    assert _recovery_time(_result_with_backlogs([0] * 9 + [1] * 7 + [0] * 12)) == 1
    assert _recovery_time(_result_with_backlogs([0] * 9 + [1] * 7 + [1] * 2 + [0] * 10)) == 3
    assert _recovery_time(_result_with_backlogs([0] * 9 + [1] * 19)) is None
    assert _recovery_time(_result_with_backlogs([0] * 28)) == 0


def test_benchmark_fairness_fresh_factories_and_schemas():
    created: list[NoOrderStrategy] = []

    def factory():
        strategy = NoOrderStrategy()
        created.append(strategy)
        return strategy

    report = run_benchmark(
        {"no_order": factory},
        SimulationConfig(horizon_days=28),
        seeds=(42, 7),
    )

    assert len(created) == 4
    assert len({id(strategy) for strategy in created}) == 4
    assert list(report.summary.columns) == SUMMARY_COLUMNS
    assert list(report.daily_trajectories.columns) == TRAJECTORY_COLUMNS
    assert len(report.summary) == 4
    assert len(report.daily_trajectories) == 4 * 28
    for seed in (42, 7):
        seed_rows = report.daily_trajectories[
            report.daily_trajectories["seed"] == seed
        ]
        known = seed_rows[seed_rows["scenario_mode"] == ScenarioMode.KNOWN_SHUTDOWN.value]
        surprise = seed_rows[
            seed_rows["scenario_mode"] == ScenarioMode.SURPRISE_SHUTDOWN.value
        ]
        assert known["realized_demand"].tolist() == surprise["realized_demand"].tolist()
        assert known["backlog_start"].iloc[0] == surprise["backlog_start"].iloc[0]
    for first, second in zip(created[::2], created[1::2]):
        first_state = first.observations[0].inventory
        second_state = second.observations[0].inventory
        assert first_state.finished_goods == second_state.finished_goods
        assert first_state.components == second_state.components
        assert first_state.backlog == second_state.backlog
        assert first_state.in_transit == second_state.in_transit


def test_benchmark_cost_orders_volatility_violations_and_invariants():
    class FixedOrders:
        def propose(self, observation):
            critical = 10 if observation.day == 1 else 20 if observation.day == 2 else 0
            from resilience_ai.contracts import OrderRequest

            return ProcurementDecision(
                day=observation.day,
                orders=(
                    OrderRequest("Supplier_Critical", critical),
                    OrderRequest("Supplier_Alt", 5),
                ),
            )

    report = run_benchmark(
        {"fixed": FixedOrders},
        SimulationConfig(horizon_days=3),
        scenario_modes=(ScenarioMode.KNOWN_SHUTDOWN,),
    )
    summary = report.summary.iloc[0]
    assert summary["critical_order_quantity"] == 30
    assert summary["alternate_order_quantity"] == 15
    assert summary["accepted_order_quantity"] == 45
    assert summary["order_volatility"] == 45
    assert summary["purchase_cost_usd"] == 480.0
    assert summary["transport_cost_usd"] == 22.5
    assert summary["procurement_expenditure_usd"] == 502.5
    assert summary["total_constraint_violations"] == 0
    assert summary["capacity_violations"] == 0
    assert bool(summary["accounting_invariants_pass"])


def test_benchmark_records_capacity_violations():
    class OverCapacity:
        def propose(self, observation):
            from resilience_ai.contracts import OrderRequest

            return ProcurementDecision(
                day=observation.day,
                orders=(OrderRequest("Supplier_Alt", 101),),
            )

    report = run_benchmark(
        {"invalid": OverCapacity},
        SimulationConfig(horizon_days=1),
        scenario_modes=(ScenarioMode.KNOWN_SHUTDOWN,),
    )
    summary = report.summary.iloc[0]
    assert summary["total_constraint_violations"] == 1
    assert summary["capacity_violations"] == 1


def test_cli_writes_evaluation_outputs(tmp_path):
    output = tmp_path / "evaluation"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_evaluation.py",
            "--seed",
            "42",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "summary_metrics.csv" in completed.stdout
    assert (output / "summary_metrics.csv").is_file()
    assert (output / "daily_trajectories.csv").is_file()
    assert (output / "aggregate_metrics.csv").is_file()
    assert len(pd.read_csv(output / "summary_metrics.csv")) == 2
    assert len(pd.read_csv(output / "daily_trajectories.csv")) == 56
    assert len(pd.read_csv(output / "aggregate_metrics.csv")) == 2


def test_aggregate_metrics_calculate_means_stds_and_trial_counts():
    report = run_benchmark(
        {"no_order": NoOrderStrategy},
        SimulationConfig(horizon_days=28),
        seeds=(42, 43, 44),
    )
    aggregate = report.aggregate_metrics
    assert list(aggregate.columns) == AGGREGATE_COLUMNS
    assert len(aggregate) == 2
    assert set(aggregate["trial_count"]) == {3}

    known_trials = report.summary[
        report.summary["scenario_mode"] == ScenarioMode.KNOWN_SHUTDOWN.value
    ]
    known_aggregate = aggregate[
        aggregate["scenario_mode"] == ScenarioMode.KNOWN_SHUTDOWN.value
    ].iloc[0]
    assert known_aggregate["same_day_fill_rate_mean"] == known_trials[
        "same_day_fill_rate"
    ].mean()
    assert known_aggregate["same_day_fill_rate_std"] == known_trials[
        "same_day_fill_rate"
    ].std(ddof=0)


def test_aggregate_metrics_ignore_missing_recovery_times():
    report = run_benchmark(
        {"no_order": NoOrderStrategy},
        SimulationConfig(horizon_days=28),
        seeds=(42, 43),
    )
    report.summary.loc[0, "recovery_time_days"] = None
    report.summary.loc[2, "recovery_time_days"] = 5
    aggregate = report.aggregate_metrics
    known_aggregate = aggregate[
        aggregate["scenario_mode"] == ScenarioMode.KNOWN_SHUTDOWN.value
    ].iloc[0]
    known_trials = report.summary[
        report.summary["scenario_mode"] == ScenarioMode.KNOWN_SHUTDOWN.value
    ]
    assert known_aggregate["trial_count"] == 2
    assert known_aggregate["recovery_time_days_mean"] == known_trials[
        "recovery_time_days"
    ].mean()
    assert known_aggregate["recovery_time_days_std"] == known_trials[
        "recovery_time_days"
    ].std(ddof=0)


def test_cli_seed_defaults_and_multi_seed_trial_counts(tmp_path):
    default_output = tmp_path / "default"
    subprocess.run(
        [sys.executable, "scripts/run_evaluation.py", "--output", str(default_output)],
        check=True,
    )
    default_summary = pd.read_csv(default_output / "summary_metrics.csv")
    assert default_summary["seed"].tolist() == [42, 42]

    multi_output = tmp_path / "multi"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_evaluation.py",
            "--seeds",
            "42",
            "43",
            "44",
            "--output",
            str(multi_output),
        ],
        check=True,
    )
    multi_summary = pd.read_csv(multi_output / "summary_metrics.csv")
    aggregate = pd.read_csv(multi_output / "aggregate_metrics.csv")
    assert len(multi_summary) == 6
    assert set(multi_summary["seed"]) == {42, 43, 44}
    assert set(aggregate["trial_count"]) == {3}


def test_cli_rejects_seed_and_seeds_together(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_evaluation.py",
            "--seed",
            "42",
            "--seeds",
            "43",
            "--output",
            str(tmp_path / "invalid"),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "not allowed with argument" in completed.stderr
