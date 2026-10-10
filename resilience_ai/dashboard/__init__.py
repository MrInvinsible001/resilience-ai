"""Dashboard module for ResilienceAI supply-chain platform."""

from resilience_ai.dashboard.mock_data import (
    create_sample_demand_forecast,
    create_sample_inventory_state,
    create_sample_observation,
    create_sample_suppliers,
)

__all__ = [
    "create_sample_demand_forecast",
    "create_sample_inventory_state",
    "create_sample_observation",
    "create_sample_suppliers",
]
