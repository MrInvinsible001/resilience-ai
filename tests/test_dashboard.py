"""Unit tests for the main ResilienceAI dashboard application logic."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from resilience_ai.coordinator import CoordinationResult
from resilience_ai.dashboard.app import (
    _get_st,
    check_ortools_availability,
    get_strategy_instance,
    main,
    prepare_comparison_metrics,
    run_dashboard_coordinator,
)
from resilience_ai.dashboard.mock_data import create_sample_observation
from resilience_ai.strategies.ortools_strategy import OrToolsStrategy
from resilience_ai.strategies.rule_based import RuleBasedStrategy


def test_app_module_importable_without_executing_ui():
    """Verify that dashboard app module can be imported without launching Streamlit or executing UI code."""
    import resilience_ai.dashboard.app as app_module

    assert hasattr(app_module, "main")
    assert hasattr(app_module, "run_dashboard_coordinator")


def test_get_strategy_instance_returns_correct_types():
    """Verify strategy selection instantiates correct strategy classes."""
    st_rb = get_strategy_instance("Rule-Based")
    assert isinstance(st_rb, RuleBasedStrategy)

    st_ot = get_strategy_instance("OR-Tools")
    assert isinstance(st_ot, OrToolsStrategy)

    st_default = get_strategy_instance("Unknown")
    assert isinstance(st_default, RuleBasedStrategy)


def test_run_dashboard_coordinator_rule_based():
    """Verify run_dashboard_coordinator produces valid CoordinationResult for Rule-Based strategy."""
    obs = create_sample_observation(day=5)
    result = run_dashboard_coordinator(obs, "Rule-Based")

    assert isinstance(result, CoordinationResult)
    assert result.day == 5
    assert result.strategy_name == "RuleBasedStrategy"
    assert result.is_valid is True


def test_run_dashboard_coordinator_ortools():
    """Verify run_dashboard_coordinator produces valid CoordinationResult for OR-Tools strategy."""
    obs = create_sample_observation(day=5)
    result = run_dashboard_coordinator(obs, "OR-Tools")

    assert isinstance(result, CoordinationResult)
    assert result.day == 5
    assert result.strategy_name == "OrToolsStrategy"
    assert result.is_valid is True


def test_prepare_comparison_metrics_equivalence():
    """Verify comparison metrics correctly aggregate decisions from equivalent observations."""
    obs = create_sample_observation(day=5)
    result_rb = run_dashboard_coordinator(obs, "Rule-Based")
    result_ot = run_dashboard_coordinator(obs, "OR-Tools")

    metrics = prepare_comparison_metrics(result_rb, result_ot, obs.suppliers)

    assert "rule_based" in metrics
    assert "ortools" in metrics
    assert metrics["rule_based"]["strategy_name"] == "RuleBasedStrategy"
    assert metrics["ortools"]["strategy_name"] == "OrToolsStrategy"
    assert isinstance(metrics["cost_difference"], float)
    assert isinstance(metrics["quantity_difference"], int)


def test_check_ortools_availability_boolean():
    """Verify check_ortools_availability returns a boolean reflecting environment state."""
    avail = check_ortools_availability()
    assert isinstance(avail, bool)


def test_streamlit_missing_raises_runtime_error():
    """Verify _get_st() and main() raise RuntimeError with clear message if Streamlit is missing."""
    with patch.dict("sys.modules", {"streamlit": None}):
        with pytest.raises(RuntimeError, match="Streamlit is not installed"):
            _get_st()

        with pytest.raises(RuntimeError, match="Streamlit is not installed"):
            main()
