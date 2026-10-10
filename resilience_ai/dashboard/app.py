"""Main Streamlit application for ResilienceAI Supply Chain Resilience Dashboard."""

from __future__ import annotations

from typing import Any, Mapping

from resilience_ai.evaluation import BenchmarkReport, run_benchmark
from resilience_ai.contracts import PlanningObservation, ScenarioMode
from resilience_ai.coordinator import CoordinationResult, Coordinator
from resilience_ai.simulator import SimulationConfig
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


def run_dashboard_benchmark(
    strategy_option: str,
    scenario_mode: ScenarioMode,
    seed: int = 42,
    config: SimulationConfig | None = None,
) -> BenchmarkReport:
    """Run the real multi-day simulator for the dashboard selection.

    ``run_benchmark`` creates a common warm-start and demand mapping for every
    strategy in a trial, so comparison mode remains fair.
    """
    strategy_factories = (
        {
            "Rule-Based": RuleBasedStrategy,
            "OR-Tools": OrToolsStrategy,
        }
        if strategy_option == "Compare Both"
        else {strategy_option: lambda: get_strategy_instance(strategy_option)}
    )
    benchmark_config = config if config is not None else SimulationConfig(seed=seed)
    return run_benchmark(
        strategy_factories,
        benchmark_config,
        scenario_modes=(scenario_mode,),
        seeds=(seed,),
    )


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
    seed = st.sidebar.number_input("Simulation Seed", min_value=0, value=42, step=1)

    ortools_available = check_ortools_availability()

    if not ortools_available:
        st.warning(
            "⚠️ **OR-Tools Package Missing:** Google OR-Tools is not installed in this environment. "
            "Selecting 'OR-Tools' strategy will transparently run its validated `RuleBasedStrategy` fallback policy."
        )

    report = run_dashboard_benchmark(strategy_option, scenario_mode, int(seed))
    summary = report.summary
    trajectories = report.daily_trajectories
    config = SimulationConfig(seed=int(seed))

    st.success(
        f"Real {config.horizon_days}-day simulator results | "
        f"{scenario_mode.value.replace('_', ' ').title()} | seed {int(seed)}"
    )
    st.caption(
        f"Shutdown: {config.critical_supplier_id} capacity is 0 on days "
        f"{config.shutdown_start_day}-{config.shutdown_end_day}. "
        f"Suppliers: {config.critical_supplier_id} ({config.critical_capacity_normal}/day, "
        f"{config.critical_lead_time_days}-day lead) and "
        f"{config.alternate_supplier_id} ({config.alternate_capacity_normal}/day, "
        f"{config.alternate_lead_time_days}-day lead)."
    )

    st.header("Simulation Summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.header("Daily Trajectories")
    st.dataframe(trajectories, use_container_width=True, hide_index=True)
    st.caption(
        "Agent insight panels backed by synthetic single-day observations remain "
        "available only through the test/demo helpers in dashboard/mock_data.py; "
        "they are not mixed into these simulation results."
    )

    with st.expander("Demo-only single-day agent insights (synthetic data)"):
        st.warning(
            "This section is retained for the existing agent UI and is not part of "
            "the real multi-day simulation results above."
        )
        from resilience_ai.dashboard.components import (
            render_coordinator_audit,
            render_demand_insights,
            render_inventory_insights,
            render_logistics_insights,
            render_procurement_orders,
            render_supplier_risk_table,
        )
        from resilience_ai.dashboard.mock_data import create_sample_observation

        demo_observation = create_sample_observation(
            day=5,
            scenario_mode=scenario_mode,
        )
        demo_strategies = (
            ("Rule-Based",)
            if strategy_option != "Compare Both"
            else ("Rule-Based", "OR-Tools")
        )
        demo_results = {
            name: run_dashboard_coordinator(demo_observation, name)
            for name in demo_strategies
        }
        for name, demo_result in demo_results.items():
            st.subheader(f"{name} — Synthetic Coordinator View")
            render_procurement_orders(
                demo_result.decision,
                demo_observation.suppliers,
            )
            tabs = st.tabs(
                [
                    "Demand Agent",
                    "Inventory Agent",
                    "Supplier-Risk Agent",
                    "Logistics Agent",
                    "Coordinator Audit",
                ]
            )
            with tabs[0]:
                render_demand_insights(demo_result.demand_signal)
            with tabs[1]:
                render_inventory_insights(demo_result.inventory_signal)
            with tabs[2]:
                render_supplier_risk_table(demo_result.supplier_risk_signal)
            with tabs[3]:
                render_logistics_insights(demo_result.logistics_signal)
            with tabs[4]:
                render_coordinator_audit(demo_result)


if __name__ == "__main__":
    main()
