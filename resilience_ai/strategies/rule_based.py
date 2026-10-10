"""Rule-based procurement strategy for the ResilienceAI supply-chain platform.

Algorithm overview
------------------
1.  Demand estimate: ``obs.forecast.mean_daily_demand`` — never future realized demand.
2.  Component coverage: on-hand components + all quantities currently in transit.
    ASSUMPTION (verify with Person 1): ``obs.inventory.in_transit`` contains only
    shipments whose ``arrival_day > obs.day``.  Shipments arriving today are
    assumed already consumed by the simulator's Stage-1 Deliveries step before
    ``propose()`` is called.  If same-day arrivals are still present this strategy
    will over-estimate coverage and under-order.
3.  Base target: ``SAFETY_DAYS * mean_daily_demand``.
4.  Known-shutdown extension: when ``scenario_mode == KNOWN_SHUTDOWN`` and a
    *disclosed* disruption is approaching, the target is extended by the full
    disruption window so that inventory is pre-built before the blackout begins.
    Only notices whose ``announced_day <= obs.day`` are trusted.  Future capacity
    fields are NEVER inspected; this strategy does not infer undisclosed disruptions.
5.  Backlog heuristic: ``inventory.backlog`` represents unfulfilled *finished-goods*
    demand — not missing components.  This strategy folds backlog into the target
    *before* the ``max(0, …)`` boundary: ``shortfall = max(0, target + backlog − coverage)``.
    This prevents over-ordering when the pipeline already exceeds the target.
    CAUTION: ordering more components does not instantly clear backlog; production
    is still bounded by ``plant_capacity``.  The heuristic is documented in the
    rationale string and tested explicitly.
    ASSUMPTION (verify with Person 1): whether including backlog in the component
    target is the correct interpretation of the simulation's accounting model.
6.  Supplier ranking: deterministic — ascending landed cost (purchase + transport),
    then ascending lead time, then supplier ID lexicographically.  Only suppliers
    with ``current_daily_capacity > 0`` are eligible.
7.  Allocation: greedy fill up to each supplier's ``current_daily_capacity``.
    Quantities are always non-negative integers (rounded down from float arithmetic).
8.  The decision is always for ``obs.day``.  ``validate_decision()`` is called
    inside the tests, not here, so that the validator remains a pure harness tool.
"""

from __future__ import annotations

from resilience_ai.contracts import (
    DisruptionNotice,
    OrderRequest,
    PlanningObservation,
    ProcurementDecision,
    ScenarioMode,
    SupplierInfo,
)

# ---------------------------------------------------------------------------
# Provisional constants — document any change here and in tests
# ---------------------------------------------------------------------------

# PROVISIONAL: number of days of forward component coverage the strategy targets.
# Set equal to the longest normal supplier lead time (alternate = 3 days) so that
# the pipeline is never empty even if today's order is the last before a stockout.
# Verify with Person 1 after warm-start inventory levels are known.
SAFETY_DAYS: int = 3

# NOTE: no DISRUPTION_BUFFER_DAYS constant is defined here.
# The pre-build extension is driven entirely by DisruptionNotice.length_days
# (the actual disclosed disruption window), so a fixed fallback constant would
# be dead code.  See Step 4 in the algorithm and module docstring.


class RuleBasedStrategy:
    """Transparent, deterministic rule-based procurement strategy.

    Implements the shared ``Strategy`` protocol.  All decisions are reproducible
    for the same ``PlanningObservation`` input and produce zero constraint
    violations when validated with ``validate_decision()``.
    """

    # No mutable state — the class is stateless so instances are interchangeable.

    def propose(self, observation: PlanningObservation) -> ProcurementDecision:
        """Return procurement orders for ``observation.day``.

        Parameters
        ----------
        observation:
            Current day's planning observation provided by the simulator harness.

        Returns
        -------
        ProcurementDecision
            A feasible proposal (verified by construction to respect all
            capacity constraints).  Call ``validate_decision(decision, obs)``
            in tests to confirm zero violations.
        """
        obs = observation

        # ------------------------------------------------------------------
        # Step 1 — Demand estimate (statistical forecast only)
        # ------------------------------------------------------------------
        daily_demand_est: float = obs.forecast.mean_daily_demand
        # obs.current_day_demand is realized today's demand — useful for the
        # rationale but must NOT be used as a forward planning estimate.

        # ------------------------------------------------------------------
        # Step 2 — Current component coverage
        # ------------------------------------------------------------------
        on_hand: int = obs.inventory.components

        # Sum all in-transit shipment quantities.
        # ASSUMPTION: in_transit holds only future-arriving shipments
        # (arrival_day > obs.day).  See module docstring for the risk if wrong.
        pipeline: int = sum(s.quantity for s in obs.inventory.in_transit)

        total_coverage: int = on_hand + pipeline

        # ------------------------------------------------------------------
        # Step 3 — Base target
        # ------------------------------------------------------------------
        base_target: float = SAFETY_DAYS * daily_demand_est

        # ------------------------------------------------------------------
        # Step 4 — Known-shutdown extension
        # ------------------------------------------------------------------
        # Only inspect explicitly disclosed disruption notices in KNOWN_SHUTDOWN
        # mode.  We trust a notice only when announced_day <= obs.day.
        # We never read future_daily_capacities or infer anything in SURPRISE mode.
        extra_blackout_days: int = 0
        triggered_supplier: str = ""
        triggered_notice: DisruptionNotice | None = None

        if obs.scenario_mode == ScenarioMode.KNOWN_SHUTDOWN:
            for notice in obs.disruptions:
                if notice.announced_day > obs.day:
                    # Should be impossible — PlanningObservation enforces this
                    # for SURPRISE_SHUTDOWN; skip undisclosed notices defensively.
                    continue

                days_until_start: int = notice.start_day - obs.day

                if days_until_start <= 0:
                    # Disruption already started or is today — cannot pre-build.
                    continue

                # Consider notices whose start is within the pre-build window:
                # we want to start ordering extra stock as early as possible,
                # so any upcoming disruption within the horizon is relevant.
                # The maximum lead time bounds when orders can still arrive before
                # the shutdown.  We use the disruption length, not the constant, so
                # the calculation stays accurate for any disruption length.
                if extra_blackout_days < notice.length_days:
                    extra_blackout_days = notice.length_days
                    triggered_supplier = notice.supplier_id
                    triggered_notice = notice

        extended_target: float = base_target + extra_blackout_days * daily_demand_est

        # ------------------------------------------------------------------
        # Step 5 — Component shortfall (with backlog heuristic)
        # ------------------------------------------------------------------
        # Backlog heuristic: treat each unit of outstanding finished-goods backlog
        # as one additional unit of component coverage required.  The formula
        # folds backlog into the target *before* the max(0, …) boundary so that:
        #   • When coverage < (target + backlog): shortfall = target + backlog - coverage
        #   • When coverage ≥ (target + backlog): shortfall = 0  (no over-ordering)
        # This prevents the previous unconditional `+ backlog` from placing orders
        # when the pipeline already exceeds the target.
        # Production remains bounded by plant_capacity; ordering more components
        # does NOT immediately clear backlog.
        # ASSUMPTION: verify with Person 1 whether 1 backlog unit ↔ 1 component
        # is the correct accounting model.
        backlog: int = obs.inventory.backlog
        shortfall: int = int(max(0.0, extended_target + backlog - total_coverage))

        # ------------------------------------------------------------------
        # Step 6 — Rank available suppliers deterministically
        # ------------------------------------------------------------------
        available_suppliers: list[SupplierInfo] = [
            s for s in obs.suppliers.values() if s.current_daily_capacity > 0
        ]

        # Sort by: (landed_cost ASC, lead_time ASC, supplier_id ASC)
        # The triple-key sort is stable and purely data-driven.
        ranked: list[SupplierInfo] = sorted(
            available_suppliers,
            key=lambda s: (
                s.unit_purchase_cost + s.unit_transport_cost,
                s.lead_time_days,
                s.supplier_id,
            ),
        )

        # ------------------------------------------------------------------
        # Step 7 — Greedy allocation
        # ------------------------------------------------------------------
        orders: list[OrderRequest] = []
        remaining: int = shortfall

        for supplier in ranked:
            if remaining <= 0:
                break
            qty: int = min(remaining, supplier.current_daily_capacity)
            # qty is already a non-negative int (min of two non-negative ints)
            orders.append(OrderRequest(supplier_id=supplier.supplier_id, quantity=qty))
            remaining -= qty

        # ------------------------------------------------------------------
        # Step 8 — Compose rationale
        # ------------------------------------------------------------------
        total_ordered: int = sum(o.quantity for o in orders)
        order_summary: str = (
            ", ".join(f"{o.supplier_id}={o.quantity}" for o in orders)
            if orders
            else "none"
        )

        disruption_note: str = ""
        if triggered_notice is not None:
            disruption_note = (
                f" | pre-build for {triggered_supplier} shutdown "
                f"days {triggered_notice.start_day}–{triggered_notice.end_day} "
                f"(+{extra_blackout_days}d buffer)"
            )

        backlog_note: str = f" | backlog={backlog} folded into target" if backlog > 0 else ""

        coverage_note: str = (
            "ADEQUATE" if shortfall == 0 else f"SHORTFALL={shortfall}"
        )

        rationale: str = (
            f"day={obs.day} mode={obs.scenario_mode.value} | "
            f"demand_today={obs.current_day_demand} demand_est={daily_demand_est:.1f}/day | "
            f"on_hand={on_hand} pipeline={pipeline} coverage={total_coverage} | "
            f"base_target={base_target:.0f} extended_target={extended_target:.0f} | "
            f"coverage={coverage_note}{disruption_note}{backlog_note} | "
            f"ordered={total_ordered} [{order_summary}] | "
            f"available_suppliers={len(available_suppliers)} ranked_suppliers={len(ranked)}"
        )

        return ProcurementDecision(day=obs.day, orders=orders, rationale=rationale)
