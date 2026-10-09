"""Reproducible strategy evaluation and comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Callable, Mapping, Sequence

import pandas as pd

from resilience_ai.contracts import ScenarioMode, Strategy, ViolationCode
from resilience_ai.simulator import (
    SimulationConfig,
    SimulationResult,
    SimulationStepLog,
    Simulator,
    generate_daily_demands,
    run_burn_in,
)


StrategyFactory = Callable[[], Strategy]

SUMMARY_COLUMNS = [
    "strategy",
    "scenario_mode",
    "seed",
    "total_realized_demand",
    "same_day_fill_rate",
    "demand_satisfied_units",
    "demand_satisfied_fraction",
    "ending_backlog",
    "peak_backlog",
    "days_with_positive_backlog",
    "cumulative_backlog_unit_days",
    "recovery_time_days",
    "purchase_cost_usd",
    "transport_cost_usd",
    "procurement_expenditure_usd",
    "critical_order_quantity",
    "alternate_order_quantity",
    "accepted_order_quantity",
    "order_volatility",
    "total_constraint_violations",
    "capacity_violations",
    "accounting_invariants_pass",
    "runtime_ms",
]

TRAJECTORY_COLUMNS = [
    "strategy",
    "scenario_mode",
    "seed",
    "day",
    "realized_demand",
    "backlog_start",
    "backlog_end",
    "produced",
    "fulfilled",
    "finished_inventory_end",
    "component_inventory_end",
    "component_arrivals",
    "same_day_served",
    "same_day_fill_rate",
    "critical_order_quantity",
    "alternate_order_quantity",
    "purchase_cost_usd",
    "transport_cost_usd",
    "constraint_violation_count",
]

AGGREGATE_METRICS = (
    "same_day_fill_rate",
    "demand_satisfied_fraction",
    "ending_backlog",
    "peak_backlog",
    "cumulative_backlog_unit_days",
    "recovery_time_days",
    "procurement_expenditure_usd",
    "total_constraint_violations",
)

AGGREGATE_COLUMNS = [
    "strategy",
    "scenario_mode",
    "trial_count",
    *[f"{metric}_mean" for metric in AGGREGATE_METRICS],
    *[f"{metric}_std" for metric in AGGREGATE_METRICS],
]


@dataclass(frozen=True)
class BenchmarkReport:
    """Stable tabular output from a benchmark run."""

    summary: pd.DataFrame
    daily_trajectories: pd.DataFrame

    @property
    def aggregate_metrics(self) -> pd.DataFrame:
        """Aggregate outcome metrics across seeds, excluding runtime."""
        grouped = self.summary.groupby(
            ["strategy", "scenario_mode"], sort=False, dropna=False
        )
        rows: list[dict[str, object]] = []
        for (strategy, scenario_mode), group in grouped:
            row: dict[str, object] = {
                "strategy": strategy,
                "scenario_mode": scenario_mode,
                "trial_count": len(group),
            }
            for metric in AGGREGATE_METRICS:
                values = group[metric]
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std(ddof=0)
            rows.append(row)
        return pd.DataFrame(rows, columns=AGGREGATE_COLUMNS)


def _order_quantities(step: SimulationStepLog, config: SimulationConfig) -> tuple[int, int]:
    quantities = {order.supplier_id: order.quantity for order in step.accepted_orders}
    return (
        quantities.get(config.critical_supplier_id, 0),
        quantities.get(config.alternate_supplier_id, 0),
    )


def _same_day_served(step: SimulationStepLog) -> int:
    return min(step.demand, max(0, step.fulfilled - step.backlog_start))


def _recovery_time(result: SimulationResult) -> int | None:
    """Return days after day 16 until backlog recovers and stays recovered."""
    backlogs = {step.day: step.backlog_end for step in result.steps}
    config = result.config
    baseline = backlogs.get(config.shutdown_start_day - 1)
    if baseline is None:
        return None
    if not any(
        backlogs.get(day, baseline) > baseline
        for day in range(config.shutdown_start_day, config.shutdown_end_day + 1)
    ):
        return 0
    for day in range(config.shutdown_end_day + 1, config.horizon_days + 1):
        if backlogs.get(day, baseline) <= baseline and all(
            backlogs.get(later, baseline) <= baseline
            for later in range(day, config.horizon_days + 1)
        ):
            return day - config.shutdown_end_day
    return None


def _build_trajectory(
    strategy_name: str,
    scenario_mode: ScenarioMode,
    seed: int,
    result: SimulationResult,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in result.steps:
        critical_qty, alternate_qty = _order_quantities(step, result.config)
        same_day_served = _same_day_served(step)
        rows.append(
            {
                "strategy": strategy_name,
                "scenario_mode": scenario_mode.value,
                "seed": seed,
                "day": step.day,
                "realized_demand": step.demand,
                "backlog_start": step.backlog_start,
                "backlog_end": step.backlog_end,
                "produced": step.produced,
                "fulfilled": step.fulfilled,
                "finished_inventory_end": step.finished_inventory_end,
                "component_inventory_end": step.components_end,
                "component_arrivals": step.arrivals,
                "same_day_served": same_day_served,
                "same_day_fill_rate": (
                    same_day_served / step.demand if step.demand else 1.0
                ),
                "critical_order_quantity": critical_qty,
                "alternate_order_quantity": alternate_qty,
                "purchase_cost_usd": step.purchase_cost,
                "transport_cost_usd": step.transport_cost,
                "constraint_violation_count": len(step.violations),
            }
        )
    return rows


def _build_summary(
    strategy_name: str,
    scenario_mode: ScenarioMode,
    seed: int,
    result: SimulationResult,
    runtime_ms: float,
) -> dict[str, object]:
    trajectory = _build_trajectory(strategy_name, scenario_mode, seed, result)
    total_demand = result.total_demand
    same_day_served = sum(int(row["same_day_served"]) for row in trajectory)
    daily_orders = [
        int(row["critical_order_quantity"]) + int(row["alternate_order_quantity"])
        for row in trajectory
    ]
    volatility = sum(
        abs(current - previous)
        for previous, current in zip([0, *daily_orders], daily_orders)
    )
    satisfied_units = max(0, total_demand - result.final_inventory.backlog)
    invariants = result.verify_invariants()
    return {
        "strategy": strategy_name,
        "scenario_mode": scenario_mode.value,
        "seed": seed,
        "total_realized_demand": total_demand,
        "same_day_fill_rate": same_day_served / total_demand if total_demand else 1.0,
        "demand_satisfied_units": satisfied_units,
        "demand_satisfied_fraction": satisfied_units / total_demand if total_demand else 1.0,
        "ending_backlog": result.final_inventory.backlog,
        "peak_backlog": max((step.backlog_end for step in result.steps), default=0),
        "days_with_positive_backlog": sum(step.backlog_end > 0 for step in result.steps),
        "cumulative_backlog_unit_days": sum(step.backlog_end for step in result.steps),
        "recovery_time_days": _recovery_time(result),
        "purchase_cost_usd": result.total_purchase_cost,
        "transport_cost_usd": result.total_transport_cost,
        "procurement_expenditure_usd": result.total_cost,
        "critical_order_quantity": sum(
            int(row["critical_order_quantity"]) for row in trajectory
        ),
        "alternate_order_quantity": sum(
            int(row["alternate_order_quantity"]) for row in trajectory
        ),
        "accepted_order_quantity": sum(daily_orders),
        "order_volatility": volatility,
        "total_constraint_violations": result.total_violations,
        "capacity_violations": sum(
            violation.code == ViolationCode.CAPACITY_EXCEEDED.value
            for violation in result.all_violations
        ),
        "accounting_invariants_pass": all(invariants.values()),
        "runtime_ms": runtime_ms,
    }


def run_benchmark(
    strategy_factories: Mapping[str, StrategyFactory],
    config: SimulationConfig,
    scenario_modes: Sequence[ScenarioMode] = (
        ScenarioMode.KNOWN_SHUTDOWN,
        ScenarioMode.SURPRISE_SHUTDOWN,
    ),
    seeds: Sequence[int] = (42,),
) -> BenchmarkReport:
    """Run each fresh strategy factory against common trials and return tables."""
    summary_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []

    for seed in seeds:
        trial_config = replace(config, seed=seed)
        initial_state, _ = run_burn_in(trial_config)
        daily_demands = generate_daily_demands(trial_config)
        for scenario_mode in scenario_modes:
            for strategy_name, factory in strategy_factories.items():
                strategy = factory()
                simulator = Simulator(
                    config=trial_config,
                    initial_state=initial_state,
                    daily_demands=daily_demands,
                )
                started = perf_counter()
                result = simulator.run(strategy, scenario_mode)
                runtime_ms = (perf_counter() - started) * 1000.0
                summary_rows.append(
                    _build_summary(
                        strategy_name, scenario_mode, seed, result, runtime_ms
                    )
                )
                trajectory_rows.extend(
                    _build_trajectory(strategy_name, scenario_mode, seed, result)
                )

    return BenchmarkReport(
        summary=pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS),
        daily_trajectories=pd.DataFrame(
            trajectory_rows, columns=TRAJECTORY_COLUMNS
        ),
    )
