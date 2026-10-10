"""Reusable UI component formatters and renderers for ResilienceAI Streamlit dashboard.

This module separates pure Python data transformation and formatting logic from
Streamlit rendering logic. Pure formatting functions can be called and unit tested
without requiring Streamlit or OR-Tools to be installed.
"""

from __future__ import annotations

from typing import Any, Mapping

from resilience_ai.agents.demand import DemandSignal
from resilience_ai.agents.inventory import InventorySignal
from resilience_ai.agents.logistics import LogisticsSignal
from resilience_ai.agents.supplier_risk import SupplierRiskSignal
from resilience_ai.contracts import (
    PlanningObservation,
    ProcurementDecision,
    SupplierInfo,
)
from resilience_ai.coordinator import CoordinationResult


def _get_st():
    """Lazily import Streamlit or raise RuntimeError if missing."""
    try:
        import streamlit as st

        return st
    except (ImportError, ModuleNotFoundError) as err:
        raise RuntimeError(
            "Streamlit is not installed in the current Python environment. "
            "Please install Streamlit (`pip install streamlit`) to render UI components."
        ) from err


# ==============================================================================
# 1. Operational Overview
# ==============================================================================


def format_operational_overview(
    observation: PlanningObservation,
    result: CoordinationResult | None = None,
) -> dict[str, Any]:
    """Transform observation and coordination result into overview metrics."""
    obs = observation
    on_hand = obs.inventory.components
    in_transit_sum = sum(s.quantity for s in obs.inventory.in_transit)
    total_coverage = on_hand + in_transit_sum

    if result is not None:
        shortfall = result.inventory_signal.component_shortfall
        is_valid = result.is_valid
        violation_count = len(result.violations)
        strategy_name = result.strategy_name
    else:
        # Standalone estimation fallback when no strategy has run
        target = obs.forecast.mean_daily_demand * 3 + obs.inventory.backlog
        shortfall = max(0, int(round(target - total_coverage)))
        is_valid = True
        violation_count = 0
        strategy_name = "None"

    return {
        "day": obs.day,
        "scenario_mode": obs.scenario_mode.value,
        "daily_demand": float(obs.forecast.mean_daily_demand),
        "on_hand_components": on_hand,
        "in_transit_components": in_transit_sum,
        "total_coverage": total_coverage,
        "backlog": obs.inventory.backlog,
        "finished_goods": obs.inventory.finished_goods,
        "shortfall": shortfall,
        "is_valid": is_valid,
        "violation_count": violation_count,
        "strategy_name": strategy_name,
    }


def render_operational_overview(
    observation: PlanningObservation,
    result: CoordinationResult | None = None,
) -> None:
    """Render operational overview KPI metrics in Streamlit."""
    st = _get_st()
    data = format_operational_overview(observation, result)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Simulation Day", f"Day {data['day']}")
    col2.metric("Scenario Mode", data['scenario_mode'].replace("_", " ").title())
    col3.metric("Daily Demand", f"{data['daily_demand']:.1f}")
    col4.metric("Total Coverage", f"{data['total_coverage']} units")
    col5.metric("Net Shortfall", f"{data['shortfall']} units")


# ==============================================================================
# 2. Procurement Orders
# ==============================================================================


def format_procurement_orders(
    decision: ProcurementDecision,
    suppliers: Mapping[str, SupplierInfo] | None = None,
) -> list[dict[str, Any]]:
    """Format procurement decision orders into structured table records.

    Handles missing supplier metadata gracefully by emitting None for unknown fields.
    """
    formatted: list[dict[str, Any]] = []

    for order in decision.orders:
        supp_id = order.supplier_id
        qty = order.quantity

        supp = suppliers.get(supp_id) if suppliers is not None else None

        if supp is not None:
            lead_time = supp.lead_time_days
            purchase_cost = float(supp.unit_purchase_cost)
            transport_cost = float(supp.unit_transport_cost)
            unit_landed = purchase_cost + transport_cost
            total_landed = unit_landed * qty
        else:
            lead_time = None
            purchase_cost = None
            transport_cost = None
            unit_landed = None
            total_landed = None

        formatted.append(
            {
                "supplier_id": supp_id,
                "quantity": qty,
                "lead_time_days": lead_time,
                "unit_purchase_cost": purchase_cost,
                "unit_transport_cost": transport_cost,
                "unit_landed_cost": unit_landed,
                "total_landed_cost": total_landed,
            }
        )

    return formatted


def calculate_decision_summary(orders_formatted: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate aggregate summary metrics for formatted orders."""
    total_orders = len(orders_formatted)
    total_qty = sum(o["quantity"] for o in orders_formatted)

    landed_costs = [o["total_landed_cost"] for o in orders_formatted if o["total_landed_cost"] is not None]
    total_landed_cost = sum(landed_costs) if landed_costs else 0.0

    return {
        "total_orders": total_orders,
        "total_quantity": total_qty,
        "total_landed_cost": total_landed_cost,
        "has_missing_costs": len(landed_costs) < total_orders,
    }


def render_procurement_orders(
    decision: ProcurementDecision,
    suppliers: Mapping[str, SupplierInfo] | None = None,
) -> None:
    """Render procurement decision orders and rationale in Streamlit."""
    st = _get_st()
    orders_data = format_procurement_orders(decision, suppliers)
    summary = calculate_decision_summary(orders_data)

    st.subheader("Procurement Decision Proposals")
    st.write(f"**Rationale:** {decision.rationale}")

    if not orders_data:
        st.info("No orders placed for this planning step.")
        return

    st.dataframe(orders_data)
    st.metric("Total Landed Cost", f"${summary['total_landed_cost']:,.2f}")


# ==============================================================================
# 3. Demand Insights
# ==============================================================================


def format_demand_insights(signal: DemandSignal) -> dict[str, Any]:
    """Transform DemandSignal into formatted dictionary for display."""
    return {
        "day": signal.day,
        "expected_daily_demand": float(signal.expected_daily_demand),
        "demand_std_dev": float(signal.demand_std_dev),
        "horizon_days": signal.horizon_days,
        "projected_total_demand": float(signal.projected_total_demand),
        "current_day_realized": signal.current_day_realized,
        "demand_deviation_today": float(signal.demand_deviation_today),
    }


def render_demand_insights(signal: DemandSignal) -> None:
    """Render Demand Agent insights in Streamlit."""
    st = _get_st()
    data = format_demand_insights(signal)

    st.subheader("Demand Agent Insights")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected Daily Demand", f"{data['expected_daily_demand']:.1f}")
    c2.metric("Demand Std Dev", f"{data['demand_std_dev']:.2f}")
    c3.metric("Realized Demand Today", f"{data['current_day_realized']}")
    c4.metric("Demand Deviation Today", f"{data['demand_deviation_today']:+.1f}")

    st.caption(f"Horizon Projection ({data['horizon_days']} days): {data['projected_total_demand']:.1f} units")


# ==============================================================================
# 4. Inventory Insights
# ==============================================================================


def format_inventory_insights(signal: InventorySignal) -> dict[str, Any]:
    """Transform InventorySignal into formatted dictionary for display."""
    return {
        "day": signal.day,
        "on_hand_components": signal.on_hand_components,
        "in_transit_components": signal.in_transit_components,
        "total_component_coverage": signal.total_component_coverage,
        "finished_goods": signal.finished_goods,
        "backlog": signal.backlog,
        "target_coverage_days": float(signal.target_coverage_days),
        "required_component_target": float(signal.required_component_target),
        "component_shortfall": signal.component_shortfall,
    }


def render_inventory_insights(signal: InventorySignal) -> None:
    """Render Inventory Agent insights in Streamlit."""
    st = _get_st()
    data = format_inventory_insights(signal)

    st.subheader("Inventory Agent Insights")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("On-Hand Components", f"{data['on_hand_components']}")
    c2.metric("In-Transit Components", f"{data['in_transit_components']}")
    c3.metric("Required Target", f"{data['required_component_target']:.1f} ({data['target_coverage_days']:.1f} days)")
    c4.metric("Component Shortfall", f"{data['component_shortfall']} units")


# ==============================================================================
# 5. Supplier-Risk Table
# ==============================================================================


def format_supplier_risk_table(signal: SupplierRiskSignal) -> list[dict[str, Any]]:
    """Transform SupplierRiskSignal into table records."""
    records: list[dict[str, Any]] = []

    for supp_id, status in signal.suppliers.items():
        records.append(
            {
                "supplier_id": status.supplier_id,
                "is_available": status.is_available,
                "current_daily_capacity": status.current_daily_capacity,
                "normal_daily_capacity": status.normal_daily_capacity,
                "capacity_ratio": float(status.capacity_ratio),
                "lead_time_days": status.lead_time_days,
                "unit_purchase_cost": float(status.unit_purchase_cost),
                "unit_transport_cost": float(status.unit_transport_cost),
                "unit_landed_cost": float(status.unit_landed_cost),
                "risk_level": status.risk_level,
                "has_active_disruption": status.has_active_disruption,
                "has_upcoming_disruption": status.has_upcoming_disruption,
                "days_until_disruption": status.days_until_disruption,
                "disruption_length_days": status.disruption_length_days,
            }
        )

    return records


def render_supplier_risk_table(signal: SupplierRiskSignal) -> None:
    """Render Supplier-Risk Agent table in Streamlit."""
    st = _get_st()
    records = format_supplier_risk_table(signal)

    st.subheader("Supplier-Risk Agent Evaluation")
    if not records:
        st.info("No suppliers evaluated.")
        return

    st.dataframe(records)
    st.write(f"**Ranked Available Suppliers:** {', '.join(signal.ranked_available_suppliers)}")


# ==============================================================================
# 6. Logistics Insights
# ==============================================================================


def format_logistics_insights(signal: LogisticsSignal) -> dict[str, Any]:
    """Transform LogisticsSignal into formatted dictionary for display."""
    return {
        "day": signal.day,
        "total_inbound_shipments": signal.total_inbound_shipments,
        "total_inbound_quantity": signal.total_inbound_quantity,
        "arriving_today_quantity": signal.arriving_today_quantity,
        "near_term_arrivals_quantity": signal.near_term_arrivals_quantity,
        "overdue_shipments_count": signal.overdue_shipments_count,
        "overdue_quantity": signal.overdue_quantity,
        "arrivals_by_day": dict(signal.arrivals_by_day),
        "supplier_inbound_quantities": dict(signal.supplier_inbound_quantities),
    }


def render_logistics_insights(signal: LogisticsSignal) -> None:
    """Render Logistics Agent insights in Streamlit."""
    st = _get_st()
    data = format_logistics_insights(signal)

    st.subheader("Logistics Agent Insights")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inbound Shipments", f"{data['total_inbound_shipments']}")
    c2.metric("Inbound Quantity", f"{data['total_inbound_quantity']}")
    c3.metric("Arriving Today", f"{data['arriving_today_quantity']}")
    c4.metric("Overdue Quantity", f"{data['overdue_quantity']}")


# ==============================================================================
# 7. Coordinator Audit
# ==============================================================================


def format_coordinator_audit(result: CoordinationResult) -> dict[str, Any]:
    """Transform CoordinationResult into audit summary."""
    violations_formatted = [
        {
            "code": v.code.value,
            "supplier_id": v.supplier_id,
            "message": v.message,
        }
        for v in result.violations
    ]

    return {
        "day": result.day,
        "strategy_name": result.strategy_name,
        "is_valid": result.is_valid,
        "rationale": result.decision.rationale,
        "violations": violations_formatted,
        "violation_count": len(violations_formatted),
    }


def render_coordinator_audit(result: CoordinationResult) -> None:
    """Render Coordinator validation audit in Streamlit."""
    st = _get_st()
    data = format_coordinator_audit(result)

    st.subheader("Coordinator Audit")
    st.write(f"**Selected Strategy:** {data['strategy_name']}")
    st.write(f"**Validation Status:** {'VALID' if data['is_valid'] else 'INVALID'}")

    if data["violations"]:
        st.error(f"Found {data['violation_count']} Constraint Violation(s):")
        st.dataframe(data["violations"])
    else:
        st.success("Decision satisfies all system constraints.")
