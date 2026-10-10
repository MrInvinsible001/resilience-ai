"""Coordinator for orchestrating multi-agent analysis and strategy proposals in ResilienceAI."""

from __future__ import annotations

from dataclasses import dataclass

from resilience_ai.agents.demand import DemandAgent, DemandSignal
from resilience_ai.agents.inventory import InventoryAgent, InventorySignal
from resilience_ai.agents.logistics import LogisticsAgent, LogisticsSignal
from resilience_ai.agents.supplier_risk import SupplierRiskAgent, SupplierRiskSignal
from resilience_ai.contracts import (
    ConstraintViolation,
    PlanningObservation,
    ProcurementDecision,
    Strategy,
    validate_decision,
)
from resilience_ai.strategies.rule_based import RuleBasedStrategy


@dataclass(frozen=True)
class CoordinationResult:
    """Immutable result emitted by Coordinator orchestrating all agents and strategies.

    Note on Immutability:
    The dataclass is frozen and violations is stored as an immutable tuple. Signal dataclasses
    (DemandSignal, InventorySignal, SupplierRiskSignal, LogisticsSignal) and ProcurementDecision
    are also frozen dataclasses. Internal mappings within signals use MappingProxyType.
    """

    day: int
    demand_signal: DemandSignal
    inventory_signal: InventorySignal
    supplier_risk_signal: SupplierRiskSignal
    logistics_signal: LogisticsSignal
    decision: ProcurementDecision
    is_valid: bool
    violations: tuple[ConstraintViolation, ...]
    strategy_name: str


class Coordinator:
    """Stateless orchestrator that runs agents, executes a Strategy proposal, and validates decisions.

    Assumptions (verify with Person 1 / Harness Lead):
    1. Simulator Entry Point: Person 1's simulation engine invokes Coordinator.run(observation)
       on each planning step to gather sub-agent signals and validate the strategy proposal.
    2. Zero Silent Repairs: The Coordinator proposes and validates decisions without modifying or
       repairing invalid orders. If is_valid is False, the simulator harness decides rejection rules.
    """

    def __init__(
        self,
        strategy: Strategy | None = None,
        demand_agent: DemandAgent | None = None,
        inventory_agent: InventoryAgent | None = None,
        supplier_risk_agent: SupplierRiskAgent | None = None,
        logistics_agent: LogisticsAgent | None = None,
    ) -> None:
        """Initialize Coordinator with optional dependency injection for strategy and agents.

        Args:
            strategy: Strategy matching Strategy protocol (defaults to RuleBasedStrategy).
            demand_agent: DemandAgent instance (defaults to concrete DemandAgent).
            inventory_agent: InventoryAgent instance (defaults to concrete InventoryAgent).
            supplier_risk_agent: SupplierRiskAgent instance (defaults to concrete SupplierRiskAgent).
            logistics_agent: LogisticsAgent instance (defaults to concrete LogisticsAgent).
        """
        self.strategy: Strategy = (
            strategy if strategy is not None else RuleBasedStrategy()
        )
        self.demand_agent: DemandAgent = (
            demand_agent if demand_agent is not None else DemandAgent()
        )
        self.inventory_agent: InventoryAgent = (
            inventory_agent if inventory_agent is not None else InventoryAgent()
        )
        self.supplier_risk_agent: SupplierRiskAgent = (
            supplier_risk_agent
            if supplier_risk_agent is not None
            else SupplierRiskAgent()
        )
        self.logistics_agent: LogisticsAgent = (
            logistics_agent if logistics_agent is not None else LogisticsAgent()
        )

    def run(self, observation: PlanningObservation) -> CoordinationResult:
        """Execute agents and strategy proposal in deterministic order and validate the decision.

        Execution Order:
        1. Demand Agent -> DemandSignal
        2. Inventory Agent -> InventorySignal (using observation and demand_signal)
        3. Supplier-Risk Agent -> SupplierRiskSignal
        4. Logistics Agent -> LogisticsSignal
        5. Strategy.propose(observation) -> ProcurementDecision
        6. validate_decision(decision, observation) -> list[ConstraintViolation]

        Args:
            observation: Current day's PlanningObservation.

        Returns:
            CoordinationResult containing all four agent signals, decision, validation status,
            violations tuple, and strategy class name.
        """
        obs = observation

        # 1. Demand Agent
        demand_signal = self.demand_agent.analyze(obs)

        # 2. Inventory Agent (using demand_signal)
        inventory_signal = self.inventory_agent.analyze(obs, demand_signal=demand_signal)

        # 3. Supplier-Risk Agent
        supplier_risk_signal = self.supplier_risk_agent.analyze(obs)

        # 4. Logistics Agent
        logistics_signal = self.logistics_agent.analyze(obs)

        # 5. Strategy proposal
        decision = self.strategy.propose(obs)

        # 6. Contract validation
        violations_list = validate_decision(decision, obs)
        is_valid = len(violations_list) == 0

        strategy_name = self.strategy.__class__.__name__

        return CoordinationResult(
            day=obs.day,
            demand_signal=demand_signal,
            inventory_signal=inventory_signal,
            supplier_risk_signal=supplier_risk_signal,
            logistics_signal=logistics_signal,
            decision=decision,
            is_valid=is_valid,
            violations=tuple(violations_list),
            strategy_name=strategy_name,
        )
