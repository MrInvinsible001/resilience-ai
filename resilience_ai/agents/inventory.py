"""Inventory Agent for the ResilienceAI supply-chain platform."""

from __future__ import annotations

from dataclasses import dataclass

from resilience_ai.agents.demand import DemandSignal
from resilience_ai.contracts import PlanningObservation, ScenarioMode

# Default forward component coverage days target during normal operations.
SAFETY_DAYS_DEFAULT: int = 3


@dataclass(frozen=True)
class InventorySignal:
    """Standardized inventory coverage and shortfall signal emitted by InventoryAgent."""

    day: int
    on_hand_components: int
    in_transit_components: int
    total_component_coverage: int
    finished_goods: int
    backlog: int
    target_coverage_days: float
    required_component_target: float
    component_shortfall: int
    time_phased_component_coverage: int = 0
    pipeline_due_within_target: int = 0


class InventoryAgent:
    """Stateless, deterministic agent for evaluating inventory coverage and component shortfall.

    Assumptions (verify with Person 1 / Harness Lead):
    1. Bill of Materials (BOM) ratio: 1 finished-goods unit (or backlog unit) requires
       exactly 1 component unit.
    2. Arrival timing: obs.inventory.in_transit is assumed to contain only shipments
       where arrival_day > obs.day. Shipments arriving today (arrival_day == obs.day)
       are assumed already consumed into on-hand components by Stage-1 simulator delivery.
    """

    def analyze(
        self,
        observation: PlanningObservation,
        demand_signal: DemandSignal,
        safety_days: int = SAFETY_DAYS_DEFAULT,
    ) -> InventorySignal:
        """Analyze planning observation and demand signal to compute inventory coverage and shortfall.

        Args:
            observation: Current day's PlanningObservation.
            demand_signal: DemandSignal emitted by DemandAgent for the current day.
            safety_days: Base forward coverage target in days (default 3).

        Returns:
            InventorySignal containing coverage, targets, and calculated shortfall.
        """
        obs = observation
        on_hand_components = obs.inventory.components
        in_transit_components = sum(s.quantity for s in obs.inventory.in_transit)
        total_component_coverage = on_hand_components + in_transit_components

        # Pre-build target extension for disclosed disruptions in KNOWN_SHUTDOWN mode.
        # Only inspect notices where announced_day <= obs.day and start_day > obs.day.
        # Take the maximum eligible length_days (not the sum of overlapping notices).
        extra_blackout_days = 0
        if obs.scenario_mode == ScenarioMode.KNOWN_SHUTDOWN:
            for notice in obs.disruptions:
                if notice.announced_day <= obs.day and notice.start_day > obs.day:
                    if notice.length_days > extra_blackout_days:
                        extra_blackout_days = notice.length_days

        target_coverage_days = float(safety_days + extra_blackout_days)
        required_component_target = float(
            target_coverage_days * demand_signal.expected_daily_demand
        )
        target_end_day = obs.day + int(target_coverage_days)
        pipeline_due_within_target = sum(
            s.quantity
            for s in obs.inventory.in_transit
            if obs.day < s.arrival_day <= target_end_day
        )
        time_phased_component_coverage = (
            on_hand_components + pipeline_due_within_target
        )

        # Backlog heuristic: fold backlog into target before max(0, ...) boundary.
        # Shortfall = max(0, required_target + backlog - total_coverage).
        component_shortfall = int(
            max(
                0.0,
                required_component_target
                + obs.inventory.backlog
                - time_phased_component_coverage,
            )
        )

        return InventorySignal(
            day=obs.day,
            on_hand_components=on_hand_components,
            in_transit_components=in_transit_components,
            total_component_coverage=total_component_coverage,
            finished_goods=obs.inventory.finished_goods,
            backlog=obs.inventory.backlog,
            target_coverage_days=target_coverage_days,
            required_component_target=required_component_target,
            component_shortfall=component_shortfall,
            time_phased_component_coverage=time_phased_component_coverage,
            pipeline_due_within_target=pipeline_due_within_target,
        )
