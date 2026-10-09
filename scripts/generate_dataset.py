import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

# Ensure repository root is on sys.path when script is executed directly
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd

from resilience_ai.contracts import ScenarioMode
from resilience_ai.simulator import (
    BaselineReactiveStrategy,
    SimulationConfig,
    Simulator,
    run_burn_in,
)


# ---------------------------------------------------------------------------
# Schema documentation (written to metadata.json)
# ---------------------------------------------------------------------------
SCHEMA = {
    "demand.csv": {
        "description": "Daily demand per distribution centre (planning input, not simulated).",
        "columns": {
            "day": "integer 1..28",
            "distribution_center": "string ID, one of DC_1 / DC_2 / DC_3",
            "demand": "integer >= 0, units (pieces) demanded that day",
        },
    },
    "supplier_capacity.csv": {
        "description": "Daily supplier capacity. Critical supplier has 0 capacity during evaluation days 10-16.",
        "columns": {
            "day": "integer 1..28",
            "critical_capacity": "integer >= 0, max units/day from Supplier_Critical",
            "alternate_capacity": "integer >= 0, max units/day from Supplier_Alt",
        },
    },
    "supplier_capacity_baseline.csv": {
        "description": "Baseline daily supplier capacity without disruption (used for surprise shutdown planning).",
        "columns": {
            "day": "integer 1..28",
            "critical_capacity": "integer >= 0, normal max units/day from Supplier_Critical (220)",
            "alternate_capacity": "integer >= 0, normal max units/day from Supplier_Alt (100)",
        },
    },
    "disruption_events.json": {
        "description": "Disruption event schedules and parameters.",
        "fields": {
            "disruptions": "array of disruption objects (supplier_id, start_day, length_days, announced_day, capacity_during_disruption, description)",
        },
    },
    "network_params.csv": {
        "description": "Static planning parameters: costs, lead times, capacities, service constraints.",
        "columns": {
            "parameter": "string key",
            "value": "numeric value",
            "unit": "string describing the unit",
            "notes": "free-text description",
        },
    },
    "initial_conditions.json": {
        "description": "Common warm-start state at day 0 from 14-day burn-in (inventory, components, pipeline).",
        "fields": {
            "day": "integer (0)",
            "finished_inventory": "integer > 0, finished units on hand at day 0",
            "component_inventory": "integer >= 0, component units on hand at day 0",
            "backlog": "integer (0), unfulfilled demand at day 0",
            "in_transit": "array of in-transit shipment objects arriving on evaluation days",
        },
    },
    "production_orders.csv": {
        "description": (
            "Simulated daily supply-chain outcomes under reactive baseline. "
            "Each day has exactly one 'production' row and two 'order' rows "
            "(one per supplier). record_type distinguishes them."
        ),
        "columns": {
            "day": "integer 1..28",
            "record_type": "'production' or 'order'",
            "produced": "(production rows only) integer >= 0, finished units made that day",
            "inventory_end": "(production rows only) integer >= 0, finished-product inventory at day end",
            "backlog_end": "(production rows only) integer >= 0, unfulfilled cumulative demand at day end",
            "component_inventory_end": "(production rows only) component stock at day end",
            "supplier": "(order rows only) 'Supplier_Critical' or 'Supplier_Alt'",
            "order_qty": "(order rows only) integer >= 0, units ordered from supplier",
            "lead_time": "(order rows only) integer days until arrival",
            "expedited": "(order rows only) boolean, True if order was expedited",
        },
    },
}


def generate(
    seed: int = 42,
    output_dir: Path | str = "data",
    plant_capacity: int = 300,
    critical_capacity: int = 220,
    alt_capacity: int = 100,
    daily_demand_mean: float = 80.0,
    daily_demand_std: float = 12.0,
    burn_in_days: int = 14,
    shutdown_start: int = 10,
    shutdown_length: int = 7,
):
    output_path = Path(output_dir)
    burn_seq, eval_seq = np.random.SeedSequence(seed).spawn(2)
    burn_rng = np.random.default_rng(burn_seq)
    eval_rng = np.random.default_rng(eval_seq)

    days = 28
    critical_supplier = "Supplier_Critical"
    alternate_supplier = "Supplier_Alt"
    distribution_centers = ("DC_1", "DC_2", "DC_3")

    lead_time_critical = 2
    lead_time_alt = 3
    critical_purchase_cost = 10.0
    alt_purchase_cost = 12.0
    transport_cost_per_unit = 0.50
    expediting_surcharge = 50.00
    service_fill_rate_target = 0.95
    target_finished_goods_buffer = 100

    config = SimulationConfig(
        horizon_days=days,
        burn_in_days=burn_in_days,
        plant_capacity=plant_capacity,
        critical_supplier_id=critical_supplier,
        alternate_supplier_id=alternate_supplier,
        critical_capacity_normal=critical_capacity,
        critical_lead_time_days=lead_time_critical,
        critical_unit_purchase_cost=critical_purchase_cost,
        critical_unit_transport_cost=transport_cost_per_unit,
        alternate_capacity_normal=alt_capacity,
        alternate_lead_time_days=lead_time_alt,
        alternate_unit_purchase_cost=alt_purchase_cost,
        alternate_unit_transport_cost=transport_cost_per_unit,
        expediting_surcharge=expediting_surcharge,
        shutdown_start_day=shutdown_start,
        shutdown_length_days=shutdown_length,
        distribution_centers=distribution_centers,
        demand_mean_per_dc=daily_demand_mean,
        demand_std_per_dc=daily_demand_std,
        seed=seed,
        target_finished_goods_buffer=target_finished_goods_buffer,
    )

    # ------------------------------------------------------------------ #
    # 1. BURN-IN: 14 days under normal capacity to establish day 0 state #
    # ------------------------------------------------------------------ #
    initial_state, burn_in_stats = run_burn_in(config, rng=burn_rng)

    initial_conditions_data = {
        "day": 0,
        "finished_inventory": initial_state.finished_goods,
        "component_inventory": initial_state.components,
        "backlog": initial_state.backlog,
        "in_transit": [
            {
                "supplier_id": s.supplier_id,
                "quantity": s.quantity,
                "order_day": s.order_day,
                "arrival_day": s.arrival_day,
                "expedited": s.expedited,
            }
            for s in initial_state.in_transit
        ],
    }

    # ------------------------------------------------------------------ #
    # 2. PLANNING INPUT 1 – demand.csv                                   #
    # ------------------------------------------------------------------ #
    demand_records = []
    for day in range(1, days + 1):
        for dc in distribution_centers:
            demand = max(0, int(eval_rng.normal(daily_demand_mean, daily_demand_std)))
            demand_records.append({"day": day, "distribution_center": dc, "demand": demand})
    demand_df = pd.DataFrame(demand_records)

    # ------------------------------------------------------------------ #
    # 3. PLANNING INPUT 2 – supplier_capacity.csv                        #
    # ------------------------------------------------------------------ #
    critical_daily_cap = []
    alt_daily_cap = []
    for day in range(1, days + 1):
        if shutdown_start <= day < shutdown_start + shutdown_length:
            crit_cap = 0
        else:
            crit_cap = critical_capacity
        critical_daily_cap.append(crit_cap)
        alt_daily_cap.append(alt_capacity)

    supplier_capacity_df = pd.DataFrame(
        {
            "day": range(1, days + 1),
            "critical_capacity": critical_daily_cap,
            "alternate_capacity": alt_daily_cap,
        }
    )

    # ------------------------------------------------------------------ #
    # 4. PLANNING INPUT 3 – supplier_capacity_baseline.csv (Scenario B)  #
    # ------------------------------------------------------------------ #
    supplier_capacity_baseline_df = pd.DataFrame(
        {
            "day": range(1, days + 1),
            "critical_capacity": [critical_capacity] * days,
            "alternate_capacity": [alt_capacity] * days,
        }
    )

    # ------------------------------------------------------------------ #
    # 5. DISRUPTION EVENT NOTICE – disruption_events.json               #
    # ------------------------------------------------------------------ #
    disruptions_data = {
        "disruptions": [
            {
                "supplier_id": critical_supplier,
                "start_day": shutdown_start,
                "length_days": shutdown_length,
                "announced_day": shutdown_start,
                "capacity_during_disruption": 0,
                "description": f"{critical_supplier} maintenance shutdown on evaluation days {shutdown_start}–{shutdown_start + shutdown_length - 1}",
            }
        ]
    }

    # ------------------------------------------------------------------ #
    # 6. PLANNING INPUT 4 – network_params.csv (static)                  #
    # ------------------------------------------------------------------ #
    network_params = [
        (
            "plant_production_capacity_per_day",
            plant_capacity,
            "units/day",
            "Max finished units Plant_A can make per day",
        ),
        (
            "initial_finished_inventory",
            initial_state.finished_goods,
            "units",
            "Finished-product stock at day 0 (from burn-in)",
        ),
        (
            "initial_component_inventory",
            initial_state.components,
            "units",
            "Component stock at day 0 (from burn-in)",
        ),
        ("lead_time_critical_days", lead_time_critical, "days", "Lead time from Supplier_Critical to Plant_A"),
        ("lead_time_alt_days", lead_time_alt, "days", "Lead time from Supplier_Alt to Plant_A"),
        (
            "critical_supplier_capacity",
            critical_capacity,
            "units/day",
            "Max daily capacity of Supplier_Critical (when operational)",
        ),
        ("alternate_supplier_capacity", alt_capacity, "units/day", "Max daily capacity of Supplier_Alt"),
        (
            "critical_unit_purchase_cost",
            critical_purchase_cost,
            "USD/unit",
            "Unit purchase cost from Supplier_Critical",
        ),
        (
            "alternate_unit_purchase_cost",
            alt_purchase_cost,
            "USD/unit",
            "Unit purchase cost from Supplier_Alt",
        ),
        (
            "transport_cost_per_unit",
            transport_cost_per_unit,
            "USD/unit",
            "Standard transport cost per component unit shipped",
        ),
        (
            "expediting_surcharge",
            expediting_surcharge,
            "USD/shipment",
            "Extra cost charged when a shipment is expedited",
        ),
        (
            "service_fill_rate_target",
            service_fill_rate_target,
            "fraction",
            "Target fraction of daily demand fulfilled on time",
        ),
        ("shutdown_start_day", shutdown_start, "day (1-indexed)", "First day Supplier_Critical is unavailable"),
        (
            "shutdown_length_days",
            shutdown_length,
            "days",
            "Number of consecutive days the critical supplier is shut down",
        ),
        (
            "demand_mean_per_dc_per_day",
            daily_demand_mean,
            "units/day",
            "Mean demand per DC per day (Normal distribution)",
        ),
        ("demand_std_per_dc_per_day", daily_demand_std, "units/day", "Std-dev of demand per DC per day"),
    ]
    network_params_df = pd.DataFrame(network_params, columns=["parameter", "value", "unit", "notes"])

    # ------------------------------------------------------------------ #
    # 7. SIMULATED BASELINE OUTCOME – production_orders.csv               #
    # ------------------------------------------------------------------ #
    daily_demands = demand_df.groupby("day")["demand"].sum().to_dict()
    sim = Simulator(config=config, initial_state=initial_state, daily_demands=daily_demands)
    baseline_strat = BaselineReactiveStrategy(stock_multiplier=1.5)
    result = sim.run(strategy=baseline_strat, scenario_mode=ScenarioMode.KNOWN_SHUTDOWN)
    production_df = result.to_dataframe()

    # ------------------------------------------------------------------ #
    # 8. Write outputs                                                    #
    # ------------------------------------------------------------------ #
    os.makedirs(output_path, exist_ok=True)
    demand_df.to_csv(output_path / "demand.csv", index=False)
    supplier_capacity_df.to_csv(output_path / "supplier_capacity.csv", index=False)
    supplier_capacity_baseline_df.to_csv(output_path / "supplier_capacity_baseline.csv", index=False)
    with open(output_path / "disruption_events.json", "w") as f:
        json.dump(disruptions_data, f, indent=2)
    with open(output_path / "initial_conditions.json", "w") as f:
        json.dump(initial_conditions_data, f, indent=2)
    network_params_df.to_csv(output_path / "network_params.csv", index=False)
    production_df.to_csv(output_path / "production_orders.csv", index=False)

    meta = {
        "seed": seed,
        "days": days,
        "burn_in_days": burn_in_days,
        "critical_supplier_shutdown": {
            "start_day": shutdown_start,
            "length": shutdown_length,
        },
        "units": "pieces",
        "schema": SCHEMA,
        "assumptions": [
            "One component required per finished unit (1:1 BOM).",
            "Demand is independently normally distributed across DCs and days.",
            "14-day burn-in period establishes a realistic initial state with zero initial backlog.",
            "The baseline ordering policy is reactive (replenish component stock to 1.5x today's demand).",
            "Orders are split proportionally to supplier capacity on the day.",
            "Alternate supplier can always be used within its capacity; it is not shut down.",
            "Production and ordering happen within the same day; arrivals happen at the start of each day.",
            "Production accounts for existing finished goods: target = max(0, demand + backlog - finished_inventory).",
            "No transit-time variability (lead times are deterministic: Critical=2, Alt=3).",
            "Transport cost is charged per unit ordered; expediting surcharge is per shipment.",
            "Service level is measured as fill-rate (fraction of demand fulfilled on time).",
            "All costs are in USD; all quantities in pieces.",
        ],
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic 28-day supply-chain dataset.")
    parser.add_argument("--output", type=Path, default=Path("data"), help="Directory to write CSV/JSON files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    generate(args.seed, args.output)


if __name__ == "__main__":
    main()
