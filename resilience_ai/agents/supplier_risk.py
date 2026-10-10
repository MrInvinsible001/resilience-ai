"""Supplier-Risk Agent for the ResilienceAI supply-chain platform."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from resilience_ai.contracts import DisruptionNotice, PlanningObservation


@dataclass(frozen=True)
class SupplierStatus:
    """Immutable risk and capability assessment for a single supplier.

    Primary risk_level precedence:
      1. "ACTIVE_DISRUPTION": A disclosed notice is currently active (start_day <= obs.day < end_day).
      2. "UPCOMING_DISRUPTION": A disclosed notice starts in the future (start_day > obs.day).
      3. "UNAVAILABLE": current_daily_capacity == 0 (with no active disruption notice).
      4. "REDUCED_CAPACITY": 0 < current_daily_capacity < normal_daily_capacity.
      5. "NORMAL": current_daily_capacity >= normal_daily_capacity.

    Individual metadata flags (has_active_disruption, has_upcoming_disruption,
    days_until_disruption) preserve specific notice facts independently of risk_level.
    """

    supplier_id: str
    is_available: bool
    current_daily_capacity: int
    normal_daily_capacity: int
    lead_time_days: int
    unit_purchase_cost: float
    unit_transport_cost: float
    unit_landed_cost: float
    capacity_ratio: float
    risk_level: str
    has_active_disruption: bool
    has_upcoming_disruption: bool
    days_until_disruption: int | None
    disruption_length_days: int


@dataclass(frozen=True)
class SupplierRiskSignal:
    """Immutable supplier risk signal emitted by SupplierRiskAgent."""

    day: int
    suppliers: Mapping[str, SupplierStatus]
    ranked_available_suppliers: tuple[str, ...]
    active_disruptions_count: int
    upcoming_disruptions_count: int


class SupplierRiskAgent:
    """Stateless, deterministic agent for evaluating supplier capacity, landed cost, and disruption risk.

    Assumptions (verify with Person 1 / Harness Lead):
    1. Landed cost is defined as unit_purchase_cost + unit_transport_cost. Expediting surcharge
       is excluded from landed-cost calculations in the current MVP.
    2. Multiple disruption notices for the same supplier are handled deterministically by prioritizing
       currently active disruptions first, then the earliest upcoming disruption.
    """

    def analyze(self, observation: PlanningObservation) -> SupplierRiskSignal:
        """Analyze current planning observation and return a SupplierRiskSignal.

        Args:
            observation: Current day's PlanningObservation.

        Returns:
            SupplierRiskSignal summarizing supplier risk statuses and landed-cost ranking.
        """
        obs = observation
        statuses: dict[str, SupplierStatus] = {}
        active_disruptions_count = 0
        upcoming_disruptions_count = 0

        # Group disclosed disruption notices by supplier_id.
        # Strict fairness check: only process notices announced on or before obs.day.
        disruptions_by_supplier: dict[str, list[DisruptionNotice]] = {}
        for notice in obs.disruptions:
            if notice.announced_day <= obs.day:
                disruptions_by_supplier.setdefault(notice.supplier_id, []).append(notice)

        for supplier_id, s in obs.suppliers.items():
            landed_cost = float(s.unit_purchase_cost + s.unit_transport_cost)
            capacity_ratio = (
                float(s.current_daily_capacity / s.normal_daily_capacity)
                if s.normal_daily_capacity > 0
                else 0.0
            )

            supplier_notices = disruptions_by_supplier.get(supplier_id, [])

            # Active disruption: start_day <= obs.day < end_day (using notice.is_active(obs.day))
            active_notice = next((n for n in supplier_notices if n.is_active(obs.day)), None)

            # Upcoming disruption: start_day > obs.day. If multiple exist, choose earliest start_day.
            upcoming_notices = [n for n in supplier_notices if n.start_day > obs.day]
            upcoming_notice = (
                min(upcoming_notices, key=lambda n: n.start_day) if upcoming_notices else None
            )

            has_active = active_notice is not None
            has_upcoming = upcoming_notice is not None

            if has_active:
                active_disruptions_count += 1
            if has_upcoming:
                upcoming_disruptions_count += 1

            # Compute independent days_until_disruption for upcoming notice if present
            days_until = (
                (upcoming_notice.start_day - obs.day) if upcoming_notice is not None else None
            )

            # Primary risk level classification precedence:
            if has_active:
                risk_level = "ACTIVE_DISRUPTION"
                disruption_len = active_notice.length_days  # type: ignore[union-attr]
            elif has_upcoming:
                risk_level = "UPCOMING_DISRUPTION"
                disruption_len = upcoming_notice.length_days  # type: ignore[union-attr]
            elif s.current_daily_capacity == 0:
                risk_level = "UNAVAILABLE"
                disruption_len = 0
            elif s.current_daily_capacity < s.normal_daily_capacity:
                risk_level = "REDUCED_CAPACITY"
                disruption_len = 0
            else:
                risk_level = "NORMAL"
                disruption_len = 0

            statuses[supplier_id] = SupplierStatus(
                supplier_id=supplier_id,
                is_available=s.is_available,
                current_daily_capacity=s.current_daily_capacity,
                normal_daily_capacity=s.normal_daily_capacity,
                lead_time_days=s.lead_time_days,
                unit_purchase_cost=float(s.unit_purchase_cost),
                unit_transport_cost=float(s.unit_transport_cost),
                unit_landed_cost=landed_cost,
                capacity_ratio=capacity_ratio,
                risk_level=risk_level,
                has_active_disruption=has_active,
                has_upcoming_disruption=has_upcoming,
                days_until_disruption=days_until,
                disruption_length_days=disruption_len,
            )

        # Rank available suppliers (current_daily_capacity > 0) deterministically
        # by (unit_landed_cost ASC, lead_time_days ASC, supplier_id ASC)
        available_suppliers = [
            s for s in obs.suppliers.values() if s.current_daily_capacity > 0
        ]
        ranked_suppliers = tuple(
            s.supplier_id
            for s in sorted(
                available_suppliers,
                key=lambda s: (
                    s.unit_purchase_cost + s.unit_transport_cost,
                    s.lead_time_days,
                    s.supplier_id,
                ),
            )
        )

        return SupplierRiskSignal(
            day=obs.day,
            suppliers=MappingProxyType(statuses),
            ranked_available_suppliers=ranked_suppliers,
            active_disruptions_count=active_disruptions_count,
            upcoming_disruptions_count=upcoming_disruptions_count,
        )
