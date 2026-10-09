# ResilienceAI — System Architecture & Shared Contract Specification

This document defines the architectural division of labor and shared contracts enabling two developers to build ResilienceAI concurrently without merge conflicts, code duplication, or interface drift.

---

## 1. Team Ownership & Responsibilities

The codebase is strictly decoupled across a shared contract boundary:

```
+---------------------------------------------------------------------------------+
|                                 SHARED CONTRACTS                                |
|                           resilience_ai/contracts.py                            |
|       (Dataclasses, Strategy Protocol, ScenarioMode, Pure Decision Validator)   |
+---------------------------------------------------------------------------------+
          ^                                                              ^
          | Implements/Consumes                                          | Implements/Consumes
          |                                                              |
+---------------------------------------+      +-----------------------------------------+
|               PERSON 1                |      |                PERSON 2                 |
|       Engine, Accounting & Evaluation |      |     Strategies, Coordinator & Dashboard |
+---------------------------------------+      +-----------------------------------------+
| - Discrete-event simulation loop      |      | - Baseline reactive strategy            |
| - Inventory & backlog accounting      |      | - Proactive buffer & heuristic planner  |
| - Warm-start burn-in (steady state)   |      | - Optimization / MILP solver strategy   |
| - Scenario runner (Known & Surprise)  |      | - Autonomous agent & coordinator        |
| - Ground-truth metrics & benchmarks   |      | - Interactive web dashboard             |
| - Harness decision execution & check  |      | - Strategy unit tests against contracts |
+---------------------------------------+      +-----------------------------------------+
```

### Person 1: Simulator, Accounting, Scenarios & Evaluation
* **Simulator Core**: Executes the 28-day simulation clock and step lifecycle (deliveries, production, fulfillment, backlog, procurement validation).
* **Production & Inventory Accounting**: Implements the corrected net-production formula, finished-goods tracking, and component inventory conservation.
* **Warm-Start Burn-In**: Runs the 14-day pre-horizon warm-up to initialize non-zero finished goods and active pipeline without starting backlog.
* **Scenario Runner**: Exposes both the **Known-Shutdown** (primary) and **Surprise-Shutdown** (stress test) environments.
* **Harness & Evaluation Engine**: Runs registered strategies, calls the shared validator, computes cost/service metrics, and records audit logs.

### Person 2: Strategies, Optimization, Coordinator & Dashboard
* **Procurement Strategies**: Implements the `Strategy` protocol for:
  1. *Reactive Baseline*: Proportional demand-tracking rule.
  2. *Heuristic / Buffer Planner*: Safety stock buildup before known disruptions.
  3. *Optimization / MILP Planner*: Mathematically optimal order schedule over horizon.
  4. *Adaptive Agent / Coordinator*: Real-time policy adjusting to surprise disruptions.
* **Interactive Dashboard**: Visualizes inventory levels, cumulative stockouts, backlog spikes, and comparative multi-strategy metrics.
* **Contract Compliance**: Ensures all strategies generate valid `ProcurementDecision` proposals satisfying daily capacity limits.

---

## 2. The Shared Contract (`resilience_ai/contracts.py`)

Both developers program against the same immutable dataclasses and protocols:

### Core Dataclasses & Types
* **`ScenarioMode`** (`Enum`): `KNOWN_SHUTDOWN` or `SURPRISE_SHUTDOWN`.
* **`Shipment`**: Represents an active in-transit shipment with `supplier_id`, `quantity`, `order_day`, `arrival_day`, and `expedited`.
* **`InventoryState`**: Snapshot of the plant at the decision point (`finished_goods`, `components`, `backlog`, `in_transit` list).
* **`DemandForecast`**: Statistical planning parameters (`mean_daily_demand`, `std_daily_demand`, `mean_demand_per_dc`, `std_demand_per_dc`, `distribution_centers`, `horizon_days`) supplied explicitly by caller without unapproved hard-coded contract defaults.
* **`SupplierInfo`**: Operational parameters, costs, and capacity for a supplier (`lead_time_days`, `current_daily_capacity`, `normal_daily_capacity`, `unit_purchase_cost`, `unit_transport_cost`, `future_daily_capacities`).
* **`DisruptionNotice`**: Specification of a capacity disruption (`supplier_id`, `start_day`, `length_days`, `announced_day`, `capacity_during_disruption`).
* **`PlanningObservation`**: Complete observation bundle passed to a strategy for Day $d$.
* **`OrderRequest`**: Individual supplier purchase proposal (`supplier_id`, `quantity`).
* **`ProcurementDecision`**: Strategy output for Day $d$ containing `day`, `orders`, and optional `rationale`.
* **`ConstraintViolation`**: Formal diagnostic emitted by the decision validator.

### The Strategy Protocol
```python
from typing import Protocol, runtime_checkable
from resilience_ai.contracts import PlanningObservation, ProcurementDecision

@runtime_checkable
class Strategy(Protocol):
    def propose(self, observation: PlanningObservation) -> ProcurementDecision:
        """Propose procurement orders for the current simulation day."""
        ...
```

---

## 3. Daily Execution Lifecycle & Accounting

On every simulation day $d \in \{1 \dots 28\}$, Person 1's engine executes these five stages in strict order:

```
+--------------------------------------------------------------------------------+
| Day d Cycle:                                                                   |
|                                                                                |
| 1. DELIVERIES (Start of Day)                                                   |
|    - Pipeline arrivals: shipments with arrival_day == d arrive at plant.       |
|    - Available components: C_avail = C_{d-1} + Arrivals_d                      |
|                                                                                |
| 2. PRODUCTION (Harness-Owned, Corrected Formula)                               |
|    - Net target: T_d = max(0, Demand_d + Backlog_{d-1} - FinishedGoods_{d-1})  |
|    - Produced: P_d = min(PlantCapacity, C_avail, T_d)                          |
|    - Remaining components: C_d = C_avail - P_d                                 |
|                                                                                |
| 3. FULFILLMENT (Harness-Owned)                                                 |
|    - Total available goods = FinishedGoods_{d-1} + P_d                         |
|    - Fulfilled: F_d = min(Total available, Demand_d + Backlog_{d-1})           |
|    - Ending finished goods: FinishedGoods_d = Total available - F_d            |
|                                                                                |
| 4. BACKLOG UPDATE (Harness-Owned)                                              |
|    - Backlog_d = (Demand_d + Backlog_{d-1}) - F_d                              |
|                                                                                |
| 5. PROCUREMENT DECISION (Strategy-Owned, Validated by Harness)                 |
|    - Construct PlanningObservation(day=d, inventory, forecast, suppliers, ...) |
|    - Strategy.propose(observation) -> ProcurementDecision                      |
|    - validate_decision(decision, observation) -> list[ConstraintViolation]     |
|    - Harness schedules valid orders: arrival_day = d + lead_time               |
+--------------------------------------------------------------------------------+
```

---

## 4. Decision Validation & Fairness Rules

### Pure Decision Validator (`validate_decision`)
The validator checks:
1. **Day Mismatch**: `decision.day == observation.day`.
2. **Invalid Quantity**: `quantity` must be a non-negative integer (rejects negative numbers, booleans, floats, strings).
3. **Duplicate Supplier Orders**: Only one order per supplier per day.
4. **Unknown Supplier**: All order targets must exist in `observation.suppliers`.
5. **Capacity Exceeded**: `order.quantity <= supplier.current_daily_capacity`.

> [!IMPORTANT]
> **Zero Clipping Guarantee**: The validator is a pure function. It **never** silently alters, clips, or mutates a strategy's decision. If any constraint is violated, explicit `ConstraintViolation` records are returned so the simulator can handle or reject the illegal proposal transparently.

### Zero Orders & Zero Capacity Semantics
* `orders = []` (no orders placed today) is valid.
* `quantity = 0` to a supplier with normal capacity is valid.
* `quantity = 0` to a supplier during shutdown (`current_daily_capacity = 0`) is valid.
* `quantity > 0` to a supplier during shutdown (`current_daily_capacity = 0`) triggers `CAPACITY_EXCEEDED`.

### Critical Fairness Boundaries
1. **No Future Demand Leaks**: Strategies only observe realized demand for the current day (`observation.current_day_demand`). They receive statistical forecasts for the future, never future realized draws.
2. **Known-Shutdown Mode**: The planned shutdown (Days 10–16) has `announced_day = 1`, making its `DisruptionNotice` and future capacity schedule visible to strategies in advance from Day 1.
3. **Surprise-Shutdown Mode**: The disruption notice has `announced_day = 10`. `PlanningObservation` enforces that:
   - Any `DisruptionNotice` with `announced_day > day` raises a `ValueError` during construction.
   - Any `SupplierInfo.future_daily_capacities` showing reduced capacity for a future day before disclosure raises a `ValueError`.
   This ensures observations cannot leak the shutdown schedule before Day 10 through notices or supplier future capacity fields.

---

## 5. Independent Development & Testing Workflow

### For Person 1 (Simulator & Evaluation)
* Import `PlanningObservation`, `ProcurementDecision`, `validate_decision` from `resilience_ai.contracts`.
* Test the engine by plugging in a dummy mock strategy (e.g. zero-order or fixed-fraction).
* Verify accounting conservation invariant:
  $$\sum_{d=1}^{28} P_d + B_{28} = \sum_{d=1}^{28} D_d + I_{28} - I_0$$

### For Person 2 (Strategies & Agents)
* Implement `Strategy` by importing types from `resilience_ai.contracts`.
* Write isolated strategy unit tests by passing synthetic `PlanningObservation` instances.
* Test edge cases:
  - Behavior when `inventory.backlog > 0`.
  - Behavior during Day 10–16 when critical supplier capacity is 0.
  - Recovery behavior on Day 17+ when alternate orders are in-transit.
* Ensure all strategy outputs satisfy `validate_decision(decision, obs) == []`.

