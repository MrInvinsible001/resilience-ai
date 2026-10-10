"""OR-Tools optimization strategy for ResilienceAI procurement framework.

Implements the shared ``Strategy`` protocol defined in ``resilience_ai.contracts``.
Uses Google OR-Tools CP-SAT solver to compute minimum landed-cost procurement
orders with lead-time tie-breaking across available suppliers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from resilience_ai.contracts import (
    DisruptionNotice,
    OrderRequest,
    PlanningObservation,
    ProcurementDecision,
    ScenarioMode,
    Strategy,
    SupplierInfo,
    validate_decision,
)
from resilience_ai.strategies.rule_based import SAFETY_DAYS, RuleBasedStrategy

if TYPE_CHECKING:
    from ortools.sat.python.cp_model import CpModel, CpSolver

logger = logging.getLogger(__name__)


class OrToolsStrategy:
    """Mathematical optimization procurement strategy using Google OR-Tools CP-SAT.

    Implements a three-stage sequential optimization model:
    - Stage 1: Maximize total procurement volume up to required shortfall R.
    - Stage 2: Minimize scaled landed cost while fixing optimal volume Q*.
    - Stage 3: Minimize total lead-time days while fixing optimal volume Q* and optimal cost C*.

    In-Transit Pipeline Limitation:
    MVP Assumption: All shipments present in ``observation.inventory.in_transit`` are
    summed into total component coverage, regardless of their ``arrival_day``. This
    matches InventoryAgent and RuleBasedStrategy behavior. If shipments arrive after the
    forward target horizon, they are still counted as current coverage.

    Cost Scaling Precision & Rounding Limitation:
    Landed cost = unit_purchase_cost + unit_transport_cost. Cost fields in contracts.py are
    floats without explicit decimal constraint. Landed costs are multiplied by 10,000 and rounded
    to integers using Python's round() (int(round(landed_cost * 10000))) to scale costs for the
    integer CP-SAT solver.
    Limitation: Two landed costs are treated as tied whenever int(round(landed_cost * 10000))
    produces the same integer coefficient for both (e.g. 10.00001 and 10.00003 both map to 100000).
    Conversely, costs differing by less than 0.00005 can produce different scaled integers if they
    straddle a rounding boundary (e.g. 10.00004 maps to 100000, while 10.00006 maps to 100001).
    Four-decimal scaling is an approximation heuristic, not a guarantee of exact cost ordering
    for arbitrary floating-point inputs.

    Solver Status & Fallback Policy:
    - OPTIMAL: Process solution and advance to next sequential stage. Stage 1 requires OPTIMAL
      to proceed to Stage 2.
    - FEASIBLE:
      - Stage 1 FEASIBLE: Non-optimal volume; falls back to RuleBasedStrategy.
      - Stage 2 FEASIBLE: Time limit reached with valid feasible solution. Used under a documented
        non-optimal policy (emits decision with [Non-Optimal Feasible Solver Output (Stage 2)] and
        skips Stage 3).
      - Stage 3 FEASIBLE: Emits valid non-optimal lead-time tie-break decision.
    - INFEASIBLE / MODEL_INVALID / UNKNOWN / Missing OR-Tools / Exception: Log reason and
      fallback to RuleBasedStrategy.
    - Safety Guarantee: Every decision is validated using ``validate_decision()``. If an
      OR-Tools or fallback proposal fails validation, an emergency empty decision (0 orders)
      is created, validated, and returned.
    """

    def __init__(
        self,
        time_limit_seconds: float = 2.0,
        rule_based_fallback: Strategy | None = None,
    ) -> None:
        """Initialize OrToolsStrategy with solver timeout and fallback strategy.

        Args:
            time_limit_seconds: Maximum solve time per stage in seconds.
            rule_based_fallback: Fallback strategy instance (defaults to RuleBasedStrategy).
        """
        self.time_limit_seconds = time_limit_seconds
        self._fallback = (
            rule_based_fallback
            if rule_based_fallback is not None
            else RuleBasedStrategy()
        )

    def propose(self, observation: PlanningObservation) -> ProcurementDecision:
        """Propose optimal supplier orders for observation.day.

        Args:
            observation: Current day's PlanningObservation.

        Returns:
            Validated ProcurementDecision.
        """
        obs = observation

        # ------------------------------------------------------------------
        # Step 1 — Target & Shortfall Calculation (matching RuleBasedStrategy)
        # ------------------------------------------------------------------
        daily_demand_est: float = obs.forecast.mean_daily_demand
        on_hand: int = obs.inventory.components

        # Pipeline limitation: count all in_transit shipments regardless of arrival_day
        pipeline: int = sum(s.quantity for s in obs.inventory.in_transit)
        total_coverage: int = on_hand + pipeline

        base_target: float = SAFETY_DAYS * daily_demand_est

        extra_blackout_days: int = 0
        if obs.scenario_mode == ScenarioMode.KNOWN_SHUTDOWN:
            for notice in obs.disruptions:
                if notice.announced_day > obs.day:
                    continue
                if notice.start_day - obs.day <= 0:
                    continue
                if extra_blackout_days < notice.length_days:
                    extra_blackout_days = notice.length_days

        extended_target: float = base_target + extra_blackout_days * daily_demand_est
        backlog: int = obs.inventory.backlog
        shortfall: int = int(max(0.0, extended_target + backlog - total_coverage))

        # Filter available suppliers with positive capacity today
        available_suppliers: list[SupplierInfo] = sorted(
            [s for s in obs.suppliers.values() if s.current_daily_capacity > 0],
            key=lambda s: s.supplier_id,
        )

        # Early return if no shortfall or no available suppliers
        if shortfall == 0 or not available_suppliers:
            rationale = (
                f"Day {obs.day}: Coverage {total_coverage} meets target {extended_target:.1f}. "
                "No orders required."
                if shortfall == 0
                else f"Day {obs.day}: Positive shortfall {shortfall}, but no available suppliers with positive capacity."
            )
            decision = ProcurementDecision(day=obs.day, orders=(), rationale=rationale)
            return self._validate_and_finalize(decision, obs, "Zero Order Decision")

        # ------------------------------------------------------------------
        # Step 2 — Attempt OR-Tools Three-Stage Optimization
        # ------------------------------------------------------------------
        try:
            from ortools.sat.python import cp_model  # Lazy import
        except (ImportError, ModuleNotFoundError) as err:
            logger.warning("OR-Tools not available (%s). Falling back to RuleBasedStrategy.", err)
            return self._execute_fallback(
                obs, f"[Fallback: OR-Tools missing ({err.__class__.__name__})]"
            )

        try:
            return self._solve_three_stage(obs, shortfall, available_suppliers, cp_model)
        except Exception as exc:
            logger.exception("OR-Tools solver exception: %s. Falling back to RuleBasedStrategy.", exc)
            return self._execute_fallback(obs, f"[Fallback: Solver exception ({exc})]")

    def _solve_three_stage(
        self,
        obs: PlanningObservation,
        shortfall: int,
        available_suppliers: list[SupplierInfo],
        cp_model_module: any,
    ) -> ProcurementDecision:
        """Execute three-stage sequential CP-SAT optimization."""
        cp_model = cp_model_module

        # ------------------------------------------------------------------
        # Stage 1: Maximize Procurement Volume Q* <= shortfall
        # ------------------------------------------------------------------
        model1 = cp_model.CpModel()
        x1 = {
            s.supplier_id: model1.NewIntVar(0, s.current_daily_capacity, f"x1_{s.supplier_id}")
            for s in available_suppliers
        }
        model1.Add(sum(x1.values()) <= shortfall)
        model1.Maximize(sum(x1.values()))

        solver1 = cp_model.CpSolver()
        solver1.parameters.max_time_in_seconds = self.time_limit_seconds
        status1 = solver1.Solve(model1)

        status1_name = solver1.StatusName(status1)
        if status1_name != "OPTIMAL":
            logger.warning("Stage 1 solver status: %s (non-optimal volume). Falling back to RuleBasedStrategy.", status1_name)
            return self._execute_fallback(obs, f"[Fallback: Stage 1 status {status1_name} (non-optimal volume)]")

        q_star = int(solver1.ObjectiveValue())
        if q_star == 0:
            decision = ProcurementDecision(
                day=obs.day, orders=(), rationale=f"Day {obs.day}: Stage 1 volume Q* = 0."
            )
            return self._validate_and_finalize(decision, obs, "Stage 1 Zero Volume")

        # ------------------------------------------------------------------
        # Stage 2: Minimize Scaled Landed Cost for Volume Q*
        # ------------------------------------------------------------------
        model2 = cp_model.CpModel()
        x2 = {
            s.supplier_id: model2.NewIntVar(0, s.current_daily_capacity, f"x2_{s.supplier_id}")
            for s in available_suppliers
        }
        model2.Add(sum(x2.values()) == q_star)

        # Cost scaling: landed_cost * 10000 rounded to integer
        cost_scaled = {
            s.supplier_id: int(round((s.unit_purchase_cost + s.unit_transport_cost) * 10000))
            for s in available_suppliers
        }
        model2.Minimize(sum(cost_scaled[s.supplier_id] * x2[s.supplier_id] for s in available_suppliers))

        solver2 = cp_model.CpSolver()
        solver2.parameters.max_time_in_seconds = self.time_limit_seconds
        status2 = solver2.Solve(model2)

        status2_name = solver2.StatusName(status2)
        if status2_name not in ("OPTIMAL", "FEASIBLE"):
            logger.warning("Stage 2 solver status: %s. Falling back to RuleBasedStrategy.", status2_name)
            return self._execute_fallback(obs, f"[Fallback: Stage 2 status {status2_name}]")

        stage2_is_optimal = status2_name == "OPTIMAL"
        c_star = int(solver2.ObjectiveValue())

        # If Stage 2 is FEASIBLE (non-optimal due to time limit), use Stage 2 solution under documented policy
        if not stage2_is_optimal:
            orders = [
                OrderRequest(supplier_id=s.supplier_id, quantity=int(solver2.Value(x2[s.supplier_id])))
                for s in available_suppliers
                if solver2.Value(x2[s.supplier_id]) > 0
            ]
            rationale = (
                f"Day {obs.day}: OR-Tools [Non-Optimal Feasible Solver Output (Stage 2)] "
                f"target Q*={q_star}, total_qty={sum(o.quantity for o in orders)}."
            )
            decision = ProcurementDecision(day=obs.day, orders=orders, rationale=rationale)
            return self._validate_and_finalize(decision, obs, "Stage 2 Feasible Decision")

        # ------------------------------------------------------------------
        # Stage 3: Minimize Lead Time for Volume Q* and Cost C*
        # ------------------------------------------------------------------
        model3 = cp_model.CpModel()
        x3 = {
            s.supplier_id: model3.NewIntVar(0, s.current_daily_capacity, f"x3_{s.supplier_id}")
            for s in available_suppliers
        }
        model3.Add(sum(x3.values()) == q_star)
        model3.Add(
            sum(cost_scaled[s.supplier_id] * x3[s.supplier_id] for s in available_suppliers) == c_star
        )
        model3.Minimize(sum(s.lead_time_days * x3[s.supplier_id] for s in available_suppliers))

        solver3 = cp_model.CpSolver()
        solver3.parameters.max_time_in_seconds = self.time_limit_seconds
        status3 = solver3.Solve(model3)

        status3_name = solver3.StatusName(status3)
        if status3_name not in ("OPTIMAL", "FEASIBLE"):
            logger.warning("Stage 3 solver status: %s. Falling back to RuleBasedStrategy.", status3_name)
            return self._execute_fallback(obs, f"[Fallback: Stage 3 status {status3_name}]")

        stage3_prefix = "OR-Tools Optimal" if status3_name == "OPTIMAL" else "OR-Tools Feasible (Stage 3 Lead Time)"

        orders = [
            OrderRequest(supplier_id=s.supplier_id, quantity=int(solver3.Value(x3[s.supplier_id])))
            for s in available_suppliers
            if solver3.Value(x3[s.supplier_id]) > 0
        ]

        total_ordered = sum(o.quantity for o in orders)
        rationale = (
            f"Day {obs.day}: {stage3_prefix} proposal. "
            f"Target Q*={q_star}, ordered={total_ordered} across {len(orders)} supplier(s)."
        )
        decision = ProcurementDecision(day=obs.day, orders=orders, rationale=rationale)
        return self._validate_and_finalize(decision, obs, "OR-Tools Decision")

    def _execute_fallback(self, obs: PlanningObservation, prefix: str) -> ProcurementDecision:
        """Execute RuleBasedStrategy fallback and validate result."""
        fb_decision = self._fallback.propose(obs)
        fallback_decision = ProcurementDecision(
            day=fb_decision.day,
            orders=fb_decision.orders,
            rationale=f"{prefix} {fb_decision.rationale}",
        )
        return self._validate_and_finalize(fallback_decision, obs, "Fallback Decision")

    def _validate_and_finalize(
        self,
        decision: ProcurementDecision,
        obs: PlanningObservation,
        source_label: str,
    ) -> ProcurementDecision:
        """Validate decision using validate_decision(); fallback or emergency empty on failure."""
        violations = validate_decision(decision, obs)
        if not violations:
            return decision

        logger.error(
            "Decision from '%s' failed validation with %d violations: %s. Initiating emergency fallback.",
            source_label,
            len(violations),
            violations,
        )

        # If primary failed and was not fallback, try rule-based fallback
        if "Fallback" not in source_label:
            fb_decision = self._fallback.propose(obs)
            fallback_decision = ProcurementDecision(
                day=fb_decision.day,
                orders=fb_decision.orders,
                rationale=f"[Fallback: Primary decision failed validation] {fb_decision.rationale}",
            )
            fb_violations = validate_decision(fallback_decision, obs)
            if not fb_violations:
                return fallback_decision

        # Ultimate safety guard: empty decision (0 orders)
        emergency_decision = ProcurementDecision(
            day=obs.day,
            orders=(),
            rationale=f"[Emergency Fallback] Invalid proposals rejected on day {obs.day}; 0 orders placed.",
        )
        em_violations = validate_decision(emergency_decision, obs)
        if em_violations:
            # Should be impossible by contracts design
            raise RuntimeError(f"Emergency empty decision failed validation: {em_violations}")

        return emergency_decision
