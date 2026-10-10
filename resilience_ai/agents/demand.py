"""Demand Agent for the ResilienceAI supply-chain platform."""

from __future__ import annotations

from dataclasses import dataclass

from resilience_ai.contracts import PlanningObservation


@dataclass(frozen=True)
class DemandSignal:
    """Standardized demand forecast signal emitted by the DemandAgent."""

    day: int
    expected_daily_demand: float
    demand_std_dev: float
    horizon_days: int
    projected_total_demand: float
    current_day_realized: int
    demand_deviation_today: float


class DemandAgent:
    """Stateless, deterministic agent for evaluating demand forecast signals."""

    def analyze(self, observation: PlanningObservation) -> DemandSignal:
        """Analyze current planning observation and return a DemandSignal.

        Args:
            observation: Current day's PlanningObservation.

        Returns:
            DemandSignal containing short-term and horizon demand expectations.
        """
        obs = observation
        forecast = obs.forecast

        # Use day-specific forecast if available; otherwise fall back to mean daily demand
        if obs.day in forecast.daily_expected_demand:
            expected_daily_demand = float(forecast.daily_expected_demand[obs.day])
        else:
            expected_daily_demand = float(forecast.mean_daily_demand)

        # Baseline horizon projection, not a sum of day-specific forecasts
        projected_total_demand = float(forecast.mean_daily_demand * forecast.horizon_days)

        demand_deviation_today = float(obs.current_day_demand - expected_daily_demand)

        return DemandSignal(
            day=obs.day,
            expected_daily_demand=expected_daily_demand,
            demand_std_dev=float(forecast.std_daily_demand),
            horizon_days=forecast.horizon_days,
            projected_total_demand=projected_total_demand,
            current_day_realized=obs.current_day_demand,
            demand_deviation_today=demand_deviation_today,
        )
