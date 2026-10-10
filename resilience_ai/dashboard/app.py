"""Main Streamlit application for ResilienceAI Supply Chain Resilience Dashboard."""

from __future__ import annotations

from typing import Any, Mapping

from resilience_ai.contracts import PlanningObservation, ScenarioMode
from resilience_ai.coordinator import CoordinationResult, Coordinator
from resilience_ai.dashboard.mock_data import create_sample_observation
from resilience_ai.strategies.ortools_strategy import OrToolsStrategy
from resilience_ai.strategies.rule_based import RuleBasedStrategy


def _get_st():
    """Lazily import Streamlit or raise RuntimeError if missing."""
    try:
        import streamlit as st

        return st
    except (ImportError, ModuleNotFoundError) as err:
        raise RuntimeError(
            "Streamlit is not installed in the current Python environment. "
            "Please run `streamlit run resilience_ai/dashboard/app.py` after installing Streamlit."
        ) from err


def check_ortools_availability() -> bool:
    """Check if ortools package is installed and importable."""
    try:
        import ortools  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


def get_strategy_instance(strategy_option: str):
    """Instantiate the strategy corresponding to the user selection."""
    if strategy_option == "OR-Tools":
        return OrToolsStrategy()
    return RuleBasedStrategy()


def run_dashboard_coordinator(
    observation: PlanningObservation, strategy_option: str
) -> CoordinationResult:
    """Run Coordinator with selected strategy on the observation."""
    strategy = get_strategy_instance(strategy_option)
    coordinator = Coordinator(strategy=strategy)
    return coordinator.run(observation)


def prepare_comparison_metrics(
    result_rule_based: CoordinationResult,
    result_ortools: CoordinationResult,
    suppliers: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare side-by-side comparison metrics between Rule-Based and OR-Tools decisions."""
    from resilience_ai.dashboard.components import (
        calculate_decision_summary,
        format_procurement_orders,
    )

    orders_rb = format_procurement_orders(result_rule_based.decision, suppliers)
    summary_rb = calculate_decision_summary(orders_rb)

    orders_ot = format_procurement_orders(result_ortools.decision, suppliers)
    summary_ot = calculate_decision_summary(orders_ot)

    cost_diff = summary_ot["total_landed_cost"] - summary_rb["total_landed_cost"]
    qty_diff = summary_ot["total_quantity"] - summary_rb["total_quantity"]

    return {
        "rule_based": {
            "strategy_name": result_rule_based.strategy_name,
            "is_valid": result_rule_based.is_valid,
            "orders_count": summary_rb["total_orders"],
            "total_quantity": summary_rb["total_quantity"],
            "total_landed_cost": summary_rb["total_landed_cost"],
            "rationale": result_rule_based.decision.rationale,
        },
        "ortools": {
            "strategy_name": result_ortools.strategy_name,
            "is_valid": result_ortools.is_valid,
            "orders_count": summary_ot["total_orders"],
            "total_quantity": summary_ot["total_quantity"],
            "total_landed_cost": summary_ot["total_landed_cost"],
            "rationale": result_ortools.decision.rationale,
        },
        "cost_difference": cost_diff,
        "quantity_difference": qty_diff,
        "is_ortools_cheaper": cost_diff < 0,
    }


def main() -> None:
    """Main dashboard entrypoint for Streamlit execution."""
    st = _get_st()

    st.set_page_config(
        page_title="ResilienceAI — Supply Chain Resilience Dashboard",
        page_icon="🏭",
        layout="wide",
    )

    st.title("ResilienceAI — Supply Chain Resilience Dashboard")

    # --- Sidebar Controls ---
    st.sidebar.header("Dashboard Configuration")

    scenario_label = st.sidebar.radio(
        "Scenario Mode",
        options=["Known Shutdown", "Surprise Shutdown"],
        index=0,
        help=(
            "Known Shutdown discloses upcoming disruptions in advance; "
            "Surprise Shutdown hides unannounced disruptions until arrival."
        ),
    )
    scenario_mode = (
        ScenarioMode.KNOWN_SHUTDOWN
        if scenario_label == "Known Shutdown"
        else ScenarioMode.SURPRISE_SHUTDOWN
    )

    strategy_option = st.sidebar.selectbox(
        "Strategy View",
        options=["Rule-Based", "OR-Tools", "Compare Both"],
        index=0,
    )

    st.sidebar.info(
        "**Note:** This dashboard operates on synthetic single-day PlanningObservation data "
        "for demo purposes. Multi-day simulation execution is currently offline pending simulator integration."
    )

    # --- System Status Banners ---
    ortools_available = check_ortools_availability()

    st.info(
        "ℹ️ **Demo Data Notice:** Operating on synthetic demo planning data. "
        "Multi-day historical trend tracking is unavailable until Person 1's simulation engine is connected."
    )

    if not ortools_available:
        st.warning(
            "⚠️ **OR-Tools Package Missing:** Google OR-Tools is not installed in this environment. "
            "Selecting 'OR-Tools' strategy will transparently run its validated `RuleBasedStrategy` fallback policy."
        )

    # Generate synthetic observation based on sidebar configuration
    obs = create_sample_observation(day=5, scenario_mode=scenario_mode)

    # Lazy import component renderers
    from resilience_ai.dashboard.components import (
        render_coordinator_audit,
        render_demand_insights,
        render_inventory_insights,
        render_logistics_insights,
        render_operational_overview,
        render_procurement_orders,
        render_supplier_risk_table,
    )

    # --- Execution & View Modes ---
    if strategy_option in ("Rule-Based", "OR-Tools"):
        result = run_dashboard_coordinator(obs, strategy_option)

        render_operational_overview(obs, result=result)
        st.markdown("---")

        render_procurement_orders(result.decision, obs.suppliers)
        st.markdown("---")

        st.header("Agent Insights & System Audit")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "Demand Agent",
                "Inventory Agent",
                "Supplier-Risk Agent",
                "Logistics Agent",
                "Coordinator Audit",
            ]
        )

        with tab1:
            render_demand_insights(result.demand_signal)
        with tab2:
            render_inventory_insights(result.inventory_signal)
        with tab3:
            render_supplier_risk_table(result.supplier_risk_signal)
        with tab4:
            render_logistics_insights(result.logistics_signal)
        with tab5:
            render_coordinator_audit(result)

    else:  # Compare Both
        result_rb = run_dashboard_coordinator(obs, "Rule-Based")
        result_ot = run_dashboard_coordinator(obs, "OR-Tools")

        render_operational_overview(obs, result=result_rb)
        st.markdown("---")

        st.header("Strategy Comparison: Rule-Based vs. OR-Tools")

        metrics = prepare_comparison_metrics(result_rb, result_ot, obs.suppliers)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Rule-Based Strategy Proposal")
            render_procurement_orders(result_rb.decision, obs.suppliers)
        with c2:
            st.subheader("OR-Tools Strategy Proposal")
            render_procurement_orders(result_ot.decision, obs.suppliers)

        st.subheader("Comparative Summary")
        cm1, cm2, cm3 = st.columns(3)
        cm1.metric("Rule-Based Total Cost", f"${metrics['rule_based']['total_landed_cost']:,.2f}")
        cm2.metric("OR-Tools Total Cost", f"${metrics['ortools']['total_landed_cost']:,.2f}")
        cm3.metric(
            "Cost Difference (OR-Tools - Rule-Based)",
            f"${metrics['cost_difference']:,.2f}",
        )

        st.markdown("---")
        st.header("Agent Insights & System Audit")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "Demand Agent",
                "Inventory Agent",
                "Supplier-Risk Agent",
                "Logistics Agent",
                "Coordinator Audit",
            ]
        )

        with tab1:
            render_demand_insights(result_rb.demand_signal)
        with tab2:
            render_inventory_insights(result_rb.inventory_signal)
        with tab3:
            render_supplier_risk_table(result_rb.supplier_risk_signal)
        with tab4:
            render_logistics_insights(result_rb.logistics_signal)
        with tab5:
            st.subheader("Coordinator Audit — Rule-Based")
            render_coordinator_audit(result_rb)
            st.markdown("---")
            st.subheader("Coordinator Audit — OR-Tools")
            render_coordinator_audit(result_ot)


if __name__ == "__main__":
    main()
