"""Procurement strategies for the ResilienceAI supply-chain platform.

Each strategy implements the shared ``Strategy`` protocol defined in
``resilience_ai.contracts`` and returns a ``ProcurementDecision`` that the
simulation harness (Person 1) can consume without needing to know which
strategy produced it.
"""

from resilience_ai.strategies.rule_based import RuleBasedStrategy

__all__ = ["RuleBasedStrategy"]
