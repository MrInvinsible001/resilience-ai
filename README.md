# ResilienceAI

A lightweight, rigorous foundation for evaluating supply-chain resilience and procurement strategies under disruptive shocks.

## Overview
This repository provides a reproducible simulation harness and synthetic dataset generator for a 28-day supply-chain network consisting of:
- **Manufacturing Plant**: Capacity of 300 units/day producing finished goods from a single component (1:1 BOM).
- **Critical Supplier** (primary): Normal capacity 220 units/day, lead time 2 days, unit purchase cost $10.00, transport cost $0.50/unit.
- **Alternate Supplier**: Capacity 100 units/day, lead time 3 days, unit purchase cost $12.00, transport cost $0.50/unit.
- **Three Distribution Centers**: Aggregate mean demand 240 units/day (80 units/day/DC, std-dev 12).
- **Critical Disruption**: A 7-day shutdown of the critical supplier on evaluation days 10–16 (capacity = 0).
- **14-Day Warm-Start Burn-In**: Eliminates artificial cold-start deficits, initializing steady-state inventory buffers and active pipeline shipments with zero starting backlog.

## Network & Planning Parameters
| Parameter | Value | Unit | Notes |
|-----------|------:|------|-------|
| Daily demand per DC (mean) | 80 | units/day | 3 DCs = 240 total units/day |
| Daily demand per DC (std-dev) | 12 | units/day | ~15% volatility |
| Plant production capacity | 300 | units/day | 25% headroom above mean demand |
| Critical supplier capacity | 220 | units/day | Days 1–9 and 17–28; 0 on days 10–16 |
| Critical supplier lead time | 2 | days | Order placed end of day $d$ arrives start of day $d+2$ |
| Alternate supplier capacity | 100 | units/day | Continuous, never disrupted |
| Alternate supplier lead time | 3 | days | Order placed end of day $d$ arrives start of day $d+3$ |
| Critical unit purchase cost | $10.00 | USD/unit | Base component purchase price |
| Alternate unit purchase cost | $12.00 | USD/unit | 20% surcharge over primary supplier |
| Unit transport cost | $0.50 | USD/unit | Standard transport fee |
| Shutdown schedule | Days 10–16 | days | 7 consecutive days |
| Evaluation horizon | 28 | days | Evaluation period |
| Burn-in period | 14 | days | Pre-evaluation warm-up under normal capacity |

## Simulation Engine Lifecycle
Every evaluation day executes in strict sequence:
1. **DELIVERIES** (Start of day): Pipeline arrivals with `arrival_day == d` are delivered into available component inventory.
2. **PRODUCTION** (Harness-owned): Plant produces the net shortfall:
   $$\text{target} = \max(0, \text{demand}_d + \text{backlog}_{d-1} - \text{finished\_inventory}_{d-1})$$
   $$\text{produced} = \min(\text{plant\_capacity}, \text{components\_available}, \text{target})$$
3. **FULFILLMENT** (Harness-owned): Fulfills demand and backlog from available finished goods:
   $$\text{fulfilled} = \min(\text{finished\_inventory}_{d-1} + \text{produced}, \text{demand}_d + \text{backlog}_{d-1})$$
4. **BACKLOG UPDATE** (Harness-owned): Unmet demand accumulates as non-negative backlog carried to the next day:
   $$\text{backlog}_d = (\text{demand}_d + \text{backlog}_{d-1}) - \text{fulfilled}$$
5. **PROCUREMENT** (Strategy-owned): Strategy receives `PlanningObservation` and submits `ProcurementDecision` proposing order quantities. Infeasible orders are rejected and diagnostic violations recorded.

## Scenarios & Information Boundaries
- **Known Shutdown** (`KNOWN_SHUTDOWN`): Advance notice and future daily capacity schedules are provided on Day 1.
- **Surprise Shutdown** (`SURPRISE_SHUTDOWN`): Advance notice is withheld until Day 10. Prior to Day 10, supplier future capacity schedules show normal capacity (220), strictly preventing advance disclosure leaks.
- **Physical Ground Truth vs Strategy Observation**: The simulator runs against physical ground truth (critical supplier capacity is 0 on Days 10–16 regardless of scenario mode). The observation layer ensures strategies only observe authorized information, preventing foresight leakage or future demand exposure.

## Parameter Rationales
- **Target Finished-Goods Buffer (100 units)**: Sized to ~42% of aggregate mean daily demand ($240$ units/day), providing ~10 hours of finished-goods stock to absorb daily stochastic demand fluctuations ($\sigma=12$ per DC) without triggering artificial backlogs during normal operations.
- **Replenishment Target Multiplier (1.5x)**: Sets the burn-in component replenishment target to $1.5 \times D_d$ ($\approx 360$ units), matching the pipeline lead-time coverage required for the 2-day primary supplier and 3-day alternate supplier.
- **Supplier Cost Differential (\$10.00 vs \$12.00)**: A 20% premium on the alternate supplier reflects emergency capacity readiness and secondary contracting costs, creating an authentic cost-resilience trade-off for optimization.

## Generating the Dataset
```bash
python scripts/generate_dataset.py --output data/ --seed 42
```
This generates all planning inputs, disruption event files, initial conditions, and reactive baseline outcomes in `data/`:
- `demand.csv`
- `supplier_capacity.csv`
- `supplier_capacity_baseline.csv`
- `disruption_events.json`
- `network_params.csv`
- `initial_conditions.json`
- `production_orders.csv`
- `metadata.json`

## Testing
Run the complete test suite:
```bash
pytest -v
```
Checks:
- All shared contracts, dataclass immutability, and decision validation.
- Dataset reproducibility, file schemas, non-negativity, and shutdown duration.
- Capacity feasibility, burn-in guards, and disruption impact.
- Simulation lead-time arrivals, capacity enforcement, conservation invariants, strategy initial-state parity, and absence of demand leaks.

## License
MIT licensed.
