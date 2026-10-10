"""Logistics Agent for the ResilienceAI supply-chain platform."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from resilience_ai.contracts import PlanningObservation

# Default horizon in days for near-term inbound shipment arrival classification.
NEAR_TERM_WINDOW_DEFAULT: int = 3


@dataclass(frozen=True)
class LogisticsSignal:
    """Immutable logistics signal emitted by LogisticsAgent summarizing inbound shipments."""

    day: int
    total_inbound_shipments: int
    total_inbound_quantity: int
    arriving_today_quantity: int
    near_term_arrivals_quantity: int
    overdue_shipments_count: int
    overdue_quantity: int
    arrivals_by_day: Mapping[int, int]
    supplier_inbound_quantities: Mapping[str, int]


class LogisticsAgent:
    """Stateless, deterministic agent for evaluating time-phased inbound logistics schedules.

    Assumptions (verify with Person 1 / Harness Lead):
    1. Same-Day Arrival Timing: If shipments with arrival_day == obs.day remain in
       obs.inventory.in_transit when propose() is called, LogisticsAgent tracks them
       in arriving_today_quantity.
    2. Overdue Retention: If a shipment misses its arrival_day, the simulator retains it in
       in_transit with arrival_day < obs.day. LogisticsAgent tracks these in overdue metrics.
    """

    def analyze(
        self,
        observation: PlanningObservation,
        near_term_window_days: int = NEAR_TERM_WINDOW_DEFAULT,
    ) -> LogisticsSignal:
        """Analyze planning observation and summarize inbound logistics arrival schedules.

        Args:
            observation: Current day's PlanningObservation.
            near_term_window_days: Horizon window (in days) for classifying near-term future arrivals.

        Returns:
            LogisticsSignal with time-phased arrivals and supplier-level inbound breakdowns.

        Raises:
            ValueError: If near_term_window_days is negative.
        """
        if near_term_window_days < 0:
            raise ValueError("near_term_window_days cannot be negative.")

        obs = observation
        in_transit = obs.inventory.in_transit

        total_shipments = len(in_transit)
        total_quantity = 0
        arriving_today_qty = 0
        near_term_qty = 0
        overdue_count = 0
        overdue_qty = 0

        raw_arrivals_by_day: dict[int, int] = {}
        raw_supplier_quantities: dict[str, int] = {}

        for s in in_transit:
            qty = s.quantity
            total_quantity += qty

            # Accumulate quantity by arrival_day
            raw_arrivals_by_day[s.arrival_day] = (
                raw_arrivals_by_day.get(s.arrival_day, 0) + qty
            )

            # Accumulate quantity by supplier_id
            raw_supplier_quantities[s.supplier_id] = (
                raw_supplier_quantities.get(s.supplier_id, 0) + qty
            )

            # Categorize arrival timing using direct arrival_day comparisons
            if s.arrival_day < obs.day:
                overdue_count += 1
                overdue_qty += qty
            elif s.arrival_day == obs.day:
                arriving_today_qty += qty
            elif obs.day < s.arrival_day <= obs.day + near_term_window_days:
                near_term_qty += qty

        # Sort keys deterministically before wrapping in MappingProxyType
        sorted_arrivals = {
            day_key: raw_arrivals_by_day[day_key]
            for day_key in sorted(raw_arrivals_by_day.keys())
        }
        sorted_suppliers = {
            supp_key: raw_supplier_quantities[supp_key]
            for supp_key in sorted(raw_supplier_quantities.keys())
        }

        return LogisticsSignal(
            day=obs.day,
            total_inbound_shipments=total_shipments,
            total_inbound_quantity=total_quantity,
            arriving_today_quantity=arriving_today_qty,
            near_term_arrivals_quantity=near_term_qty,
            overdue_shipments_count=overdue_count,
            overdue_quantity=overdue_qty,
            arrivals_by_day=MappingProxyType(sorted_arrivals),
            supplier_inbound_quantities=MappingProxyType(sorted_suppliers),
        )
