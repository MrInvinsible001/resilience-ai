"""Unit tests for SupplierRiskAgent, SupplierStatus, and SupplierRiskSignal."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from resilience_ai.agents.supplier_risk import (
    SupplierRiskAgent,
    SupplierRiskSignal,
    SupplierStatus,
)
from resilience_ai.contracts import (
    DemandForecast,
    DisruptionNotice,
    InventoryState,
    PlanningObservation,
    ScenarioMode,
    SupplierInfo,
)


def make_supplier_test_observation(
    day: int = 5,
    scenario_mode: ScenarioMode = ScenarioMode.KNOWN_SHUTDOWN,
    suppliers: dict[str, SupplierInfo] | None = None,
    disruptions: list[DisruptionNotice] | None = None,
) -> PlanningObservation:
    """Helper to construct a valid observation for SupplierRiskAgent tests."""
    if suppliers is None:
        suppliers = {
            "Supplier_Critical": SupplierInfo(
                supplier_id="Supplier_Critical",
                lead_time_days=2,
                current_daily_capacity=220,
                normal_daily_capacity=220,
                unit_purchase_cost=10.0,
                unit_transport_cost=0.50,
                expediting_surcharge=5.0,
            ),
            "Supplier_Alt": SupplierInfo(
                supplier_id="Supplier_Alt",
                lead_time_days=3,
                current_daily_capacity=100,
                normal_daily_capacity=100,
                unit_purchase_cost=12.0,
                unit_transport_cost=0.50,
                expediting_surcharge=5.0,
            ),
        }

    inventory = InventoryState(
        day=day,
        finished_goods=120,
        components=80,
        backlog=0,
        in_transit=[],
    )

    forecast = DemandForecast(
        mean_daily_demand=240.0,
        std_daily_demand=20.8,
        mean_demand_per_dc=80.0,
        std_demand_per_dc=12.0,
        distribution_centers=("DC_1", "DC_2", "DC_3"),
        horizon_days=28,
    )

    return PlanningObservation(
        day=day,
        scenario_mode=scenario_mode,
        inventory=inventory,
        current_day_demand=240,
        forecast=forecast,
        suppliers=suppliers,
        disruptions=disruptions if disruptions is not None else [],
    )


def test_supplier_risk_agent_deterministic_output():
    """Calling analyze() twice with identical observations produces identical signals."""
    obs = make_supplier_test_observation()
    agent = SupplierRiskAgent()

    signal1 = agent.analyze(obs)
    signal2 = agent.analyze(obs)

    assert signal1 == signal2
    assert signal1.day == 5
    assert len(signal1.suppliers) == 2


def test_supplier_risk_agent_landed_cost_excludes_expediting():
    """unit_landed_cost = unit_purchase_cost + unit_transport_cost, excluding expediting_surcharge."""
    obs = make_supplier_test_observation()
    agent = SupplierRiskAgent()

    signal = agent.analyze(obs)
    critical_status = signal.suppliers["Supplier_Critical"]

    # 10.0 + 0.50 = 10.50 (ignoring expediting_surcharge=5.0)
    assert critical_status.unit_landed_cost == 10.50
    assert critical_status.unit_purchase_cost == 10.0
    assert critical_status.unit_transport_cost == 0.50


def test_supplier_risk_agent_ranking_order():
    """Available suppliers are ranked by (unit_landed_cost ASC, lead_time_days ASC, supplier_id ASC)."""
    suppliers = {
        "Supplier_C": SupplierInfo(
            supplier_id="Supplier_C",
            lead_time_days=3,
            current_daily_capacity=100,
            normal_daily_capacity=100,
            unit_purchase_cost=10.0,
            unit_transport_cost=1.0,  # landed = 11.0
        ),
        "Supplier_A": SupplierInfo(
            supplier_id="Supplier_A",
            lead_time_days=2,
            current_daily_capacity=100,
            normal_daily_capacity=100,
            unit_purchase_cost=10.0,
            unit_transport_cost=0.5,  # landed = 10.5
        ),
        "Supplier_B": SupplierInfo(
            supplier_id="Supplier_B",
            lead_time_days=1,
            current_daily_capacity=100,
            normal_daily_capacity=100,
            unit_purchase_cost=10.0,
            unit_transport_cost=0.5,  # landed = 10.5 (tie on landed, smaller lead time)
        ),
    }
    obs = make_supplier_test_observation(suppliers=suppliers)
    agent = SupplierRiskAgent()

    signal = agent.analyze(obs)

    # Landed costs: B (10.5, lead 1), A (10.5, lead 2), C (11.0, lead 3)
    assert signal.ranked_available_suppliers == ("Supplier_B", "Supplier_A", "Supplier_C")


def test_supplier_risk_agent_zero_capacity_unavailable():
    """Zero-capacity supplier is marked is_available=False, UNAVAILABLE, and excluded from ranking."""
    suppliers = {
        "Supplier_Available": SupplierInfo(
            supplier_id="Supplier_Available",
            lead_time_days=2,
            current_daily_capacity=100,
            normal_daily_capacity=100,
            unit_purchase_cost=10.0,
            unit_transport_cost=1.0,
        ),
        "Supplier_ZeroCap": SupplierInfo(
            supplier_id="Supplier_ZeroCap",
            lead_time_days=2,
            current_daily_capacity=0,
            normal_daily_capacity=100,
            unit_purchase_cost=8.0,
            unit_transport_cost=0.5,
        ),
    }
    obs = make_supplier_test_observation(suppliers=suppliers)
    agent = SupplierRiskAgent()

    signal = agent.analyze(obs)
    zero_status = signal.suppliers["Supplier_ZeroCap"]

    assert zero_status.is_available is False
    assert zero_status.risk_level == "UNAVAILABLE"
    assert zero_status.capacity_ratio == 0.0
    assert signal.ranked_available_suppliers == ("Supplier_Available",)


def test_supplier_risk_agent_active_disruption():
    """Disclosed notice active on current day (start <= day < end) classified as ACTIVE_DISRUPTION."""
    notice = DisruptionNotice(
        supplier_id="Supplier_Critical",
        start_day=4,
        length_days=5,  # active days 4, 5, 6, 7, 8
        announced_day=1,
    )
    obs = make_supplier_test_observation(day=5, disruptions=[notice])
    agent = SupplierRiskAgent()

    signal = agent.analyze(obs)
    status = signal.suppliers["Supplier_Critical"]

    assert status.risk_level == "ACTIVE_DISRUPTION"
    assert status.has_active_disruption is True
    assert status.disruption_length_days == 5
    assert signal.active_disruptions_count == 1


def test_supplier_risk_agent_upcoming_disruption():
    """Disclosed notice starting in future (start > day) classified as UPCOMING_DISRUPTION with correct days_until."""
    notice = DisruptionNotice(
        supplier_id="Supplier_Critical",
        start_day=10,
        length_days=7,
        announced_day=1,  # announced day 1, today is day 5
    )
    obs = make_supplier_test_observation(day=5, disruptions=[notice])
    agent = SupplierRiskAgent()

    signal = agent.analyze(obs)
    status = signal.suppliers["Supplier_Critical"]

    assert status.risk_level == "UPCOMING_DISRUPTION"
    assert status.has_upcoming_disruption is True
    assert status.days_until_disruption == 5  # 10 - 5 = 5
    assert status.disruption_length_days == 7
    assert signal.upcoming_disruptions_count == 1


def test_supplier_risk_agent_unannounced_future_notice_ignored():
    """Notices with announced_day > obs.day are ignored by SupplierRiskAgent, and blocked in SURPRISE mode."""
    unannounced_notice = DisruptionNotice(
        supplier_id="Supplier_Critical",
        start_day=10,
        length_days=7,
        announced_day=6,  # announced tomorrow (day 6)
    )

    # In KNOWN_SHUTDOWN, unannounced_notice is ignored by SupplierRiskAgent because announced_day > day
    obs_known = make_supplier_test_observation(
        day=5, scenario_mode=ScenarioMode.KNOWN_SHUTDOWN, disruptions=[unannounced_notice]
    )
    agent = SupplierRiskAgent()
    sig_known = agent.analyze(obs_known)
    status = sig_known.suppliers["Supplier_Critical"]

    assert status.risk_level == "NORMAL"
    assert status.has_active_disruption is False
    assert status.has_upcoming_disruption is False
    assert status.days_until_disruption is None
    assert sig_known.upcoming_disruptions_count == 0

    # In SURPRISE_SHUTDOWN, PlanningObservation contract constructor actively enforces fairness
    with pytest.raises(ValueError, match="Fairness violation"):
        make_supplier_test_observation(
            day=5, scenario_mode=ScenarioMode.SURPRISE_SHUTDOWN, disruptions=[unannounced_notice]
        )


def test_supplier_risk_agent_surprise_shutdown_no_future_capacity_leak():
    """In SURPRISE_SHUTDOWN mode, supplier capacity reductions without notice are reported without predicting events."""
    suppliers = {
        "Supplier_Critical": SupplierInfo(
            supplier_id="Supplier_Critical",
            lead_time_days=2,
            current_daily_capacity=0,  # surprise drop to zero today
            normal_daily_capacity=220,
            unit_purchase_cost=10.0,
            unit_transport_cost=0.50,
            future_daily_capacities={6: 220, 7: 220},  # valid future capacities
        )
    }
    obs = make_supplier_test_observation(
        day=5, scenario_mode=ScenarioMode.SURPRISE_SHUTDOWN, suppliers=suppliers, disruptions=[]
    )
    agent = SupplierRiskAgent()

    signal = agent.analyze(obs)
    status = signal.suppliers["Supplier_Critical"]

    # Evaluates capacity today strictly without reading future_daily_capacities
    assert status.risk_level == "UNAVAILABLE"
    assert status.has_active_disruption is False
    assert status.has_upcoming_disruption is False


def test_supplier_risk_agent_reduced_capacity():
    """Supplier with 0 < current < normal capacity classified as REDUCED_CAPACITY."""
    suppliers = {
        "Supplier_Critical": SupplierInfo(
            supplier_id="Supplier_Critical",
            lead_time_days=2,
            current_daily_capacity=110,  # 50% of normal 220
            normal_daily_capacity=220,
            unit_purchase_cost=10.0,
            unit_transport_cost=0.50,
        )
    }
    obs = make_supplier_test_observation(suppliers=suppliers)
    agent = SupplierRiskAgent()

    signal = agent.analyze(obs)
    status = signal.suppliers["Supplier_Critical"]

    assert status.risk_level == "REDUCED_CAPACITY"
    assert status.capacity_ratio == 0.5
    assert status.is_available is True


def test_supplier_risk_agent_multiple_notices_handled_deterministically():
    """Multiple notices for same supplier are handled safely without duplicate counting."""
    notices = [
        DisruptionNotice(
            supplier_id="Supplier_Critical",
            start_day=4,
            length_days=3,  # active day 4..6
            announced_day=1,
        ),
        DisruptionNotice(
            supplier_id="Supplier_Critical",
            start_day=10,
            length_days=5,  # upcoming day 10..14
            announced_day=2,
        ),
    ]
    obs = make_supplier_test_observation(day=5, disruptions=notices)
    agent = SupplierRiskAgent()

    signal = agent.analyze(obs)
    status = signal.suppliers["Supplier_Critical"]

    assert status.risk_level == "ACTIVE_DISRUPTION"
    assert status.has_active_disruption is True
    assert status.has_upcoming_disruption is True
    assert status.days_until_disruption == 5  # 10 - 5
    # Each category count increments at most once per supplier
    assert signal.active_disruptions_count == 1
    assert signal.upcoming_disruptions_count == 1


def test_supplier_risk_agent_risk_metadata_visible_when_reduced_capacity():
    """Upcoming notice metadata remains visible when supplier has reduced capacity and an upcoming notice."""
    suppliers = {
        "Supplier_Critical": SupplierInfo(
            supplier_id="Supplier_Critical",
            lead_time_days=2,
            current_daily_capacity=110,  # reduced capacity today
            normal_daily_capacity=220,
            unit_purchase_cost=10.0,
            unit_transport_cost=0.50,
        )
    }
    upcoming_notice = DisruptionNotice(
        supplier_id="Supplier_Critical",
        start_day=12,
        length_days=7,
        announced_day=1,
    )
    obs = make_supplier_test_observation(day=5, suppliers=suppliers, disruptions=[upcoming_notice])
    agent = SupplierRiskAgent()

    signal = agent.analyze(obs)
    status = signal.suppliers["Supplier_Critical"]

    # Upcoming disruption takes precedence over reduced capacity in primary risk_level
    assert status.risk_level == "UPCOMING_DISRUPTION"
    assert status.has_upcoming_disruption is True
    assert status.days_until_disruption == 7  # 12 - 5
    assert status.capacity_ratio == 0.5
    assert status.current_daily_capacity == 110


def test_supplier_risk_signal_is_frozen_and_immutable():
    """SupplierRiskSignal and SupplierStatus are frozen; suppliers mapping is read-only."""
    obs = make_supplier_test_observation()
    agent = SupplierRiskAgent()

    signal = agent.analyze(obs)

    # Test dataclass immutability
    with pytest.raises(FrozenInstanceError):
        signal.day = 999  # type: ignore[misc]

    status = signal.suppliers["Supplier_Critical"]
    with pytest.raises(FrozenInstanceError):
        status.unit_landed_cost = 99.0  # type: ignore[misc]

    # Test MappingProxyType immutability
    with pytest.raises(TypeError):
        signal.suppliers["Supplier_Critical"] = status  # type: ignore[index]


def test_supplier_risk_agent_empty_suppliers():
    """Empty suppliers dict in PlanningObservation handled safely without errors."""
    obs = make_supplier_test_observation(suppliers={})
    agent = SupplierRiskAgent()

    signal = agent.analyze(obs)

    assert len(signal.suppliers) == 0
    assert signal.ranked_available_suppliers == ()
    assert signal.active_disruptions_count == 0
    assert signal.upcoming_disruptions_count == 0
