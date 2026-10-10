"""Coordinator for orchestrating multi-agent analysis and strategy proposals in ResilienceAI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from resilience_ai.agents.demand import DemandAgent, DemandSignal
from resilience_ai.agents.inventory import InventoryAgent, InventorySignal
from resilience_ai.agents.logistics import LogisticsAgent, LogisticsSignal
from resilience_ai.agents.supplier_risk import SupplierRiskAgent, SupplierRiskSignal
from resilience_ai.contracts import (
    ConstraintViolation,
    OrderRequest,
    PlanningObservation,
    ProcurementDecision,
    Strategy,
    validate_decision,
)
from resilience_ai.strategies.rule_based import RuleBasedStrategy


class ApprovalStatus(str, Enum):
    """Explicit lifecycle state for a planner recommendation."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class PlannerApproval:
    """Immutable record of a planner decision and its approval state."""

    day: int
    decision: ProcurementDecision
    status: ApprovalStatus = ApprovalStatus.PENDING
    reason: str = ""


class PlannerApprovalWorkflow:
    """Small in-memory approval workflow kept separate from physical execution."""

    def __init__(self) -> None:
        self._records: dict[int, PlannerApproval] = {}

    def submit(self, decision: ProcurementDecision) -> PlannerApproval:
        record = PlannerApproval(day=decision.day, decision=decision)
        self._records[decision.day] = record
        return record

    def approve(self, day: int, reason: str = "") -> ProcurementDecision:
        record = self._records.get(day)
        if record is None:
            raise KeyError(f"No pending recommendation exists for day {day}.")
        if record.status is not ApprovalStatus.PENDING:
            raise ValueError(f"Recommendation for day {day} is already {record.status.value}.")
        self._records[day] = PlannerApproval(
            day=day,
            decision=record.decision,
            status=ApprovalStatus.APPROVED,
            reason=reason,
        )
        return record.decision

    def reject(self, day: int, reason: str = "") -> PlannerApproval:
        record = self._records.get(day)
        if record is None:
            raise KeyError(f"No pending recommendation exists for day {day}.")
        if record.status is not ApprovalStatus.PENDING:
            raise ValueError(f"Recommendation for day {day} is already {record.status.value}.")
        rejected = PlannerApproval(
            day=day,
            decision=record.decision,
            status=ApprovalStatus.REJECTED,
            reason=reason,
        )
        self._records[day] = rejected
        return rejected

    def get(self, day: int) -> PlannerApproval | None:
        return self._records.get(day)


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
    approval_status: ApprovalStatus = ApprovalStatus.PENDING


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
        reconcile_signals: bool = False,
        approval_workflow: PlannerApprovalWorkflow | None = None,
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
        self.reconcile_signals = reconcile_signals
        self.approval_workflow = approval_workflow or PlannerApprovalWorkflow()

    def _reconcile(
        self,
        observation: PlanningObservation,
        decision: ProcurementDecision,
        inventory_signal: InventorySignal,
        supplier_risk_signal: SupplierRiskSignal,
        logistics_signal: LogisticsSignal,
    ) -> ProcurementDecision:
        """Reconcile a valid strategy proposal using all four agent outputs.

        Priorities are explicit: protect feasible component coverage, use
        disclosed/currently available suppliers ranked by landed cost, and do
        not order when time-phased inbound inventory already covers the target.
        """
        base_orders = {order.supplier_id: order.quantity for order in decision.orders}
        current_order_total = sum(base_orders.values())
        required = max(
            0,
            inventory_signal.component_shortfall
            - logistics_signal.near_term_arrivals_quantity,
        )
        additional = max(0, required - current_order_total)

        for supplier_id in supplier_risk_signal.ranked_available_suppliers:
            if additional <= 0:
                break
            supplier = observation.suppliers[supplier_id]
            already = base_orders.get(supplier_id, 0)
            room = max(0, supplier.current_daily_capacity - already)
            quantity = min(room, additional)
            if quantity:
                base_orders[supplier_id] = already + quantity
                additional -= quantity

        orders = tuple(
            OrderRequest(supplier_id=supplier_id, quantity=quantity)
            for supplier_id, quantity in base_orders.items()
            if quantity > 0
        )
        rationale = (
            f"{decision.rationale} | reconciled using demand={inventory_signal.required_component_target:.1f}, "
            f"time_phased_coverage={inventory_signal.time_phased_component_coverage}, "
            f"near_term_arrivals={logistics_signal.near_term_arrivals_quantity}, "
            f"supplier_priority={','.join(supplier_risk_signal.ranked_available_suppliers)}"
        )
        return ProcurementDecision(
            day=decision.day,
            orders=orders,
            rationale=rationale,
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

        # Invalid proposals are preserved exactly; only valid proposals are reconciled.
        initial_violations = validate_decision(decision, obs)
        if self.reconcile_signals and not initial_violations:
            decision = self._reconcile(
                obs,
                decision,
                inventory_signal,
                supplier_risk_signal,
                logistics_signal,
            )

        # 6. Contract validation
        violations_list = validate_decision(decision, obs)
        is_valid = len(violations_list) == 0
        approval = self.approval_workflow.submit(decision)

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
            approval_status=approval.status,
        )

    def approve(self, day: int, reason: str = "") -> ProcurementDecision:
        """Approve a pending recommendation for explicit physical execution."""
        return self.approval_workflow.approve(day, reason)

    def reject(self, day: int, reason: str = "") -> PlannerApproval:
        """Reject a pending recommendation without changing physical state."""
        return self.approval_workflow.reject(day, reason)


class CoordinatedStrategy:
    """Strategy adapter that connects agent reconciliation to the simulator.

    Approval is explicit in this adapter: a valid recommendation is submitted
    as pending, then approved by this named simulation policy before return.
    External planner workflows can instead call ``Coordinator.approve``.
    """

    def __init__(self, strategy: Strategy | None = None) -> None:
        self.coordinator = Coordinator(
            strategy=strategy or RuleBasedStrategy(),
            reconcile_signals=True,
        )

    def propose(self, observation: PlanningObservation) -> ProcurementDecision:
        result = self.coordinator.run(observation)
        if not result.is_valid:
            return result.decision
        return self.coordinator.approve(
            observation.day,
            reason="Approved by coordinated simulation policy",
        )
