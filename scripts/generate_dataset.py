import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd


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
        "description": "Daily supplier capacity. Critical supplier has 0 capacity during shutdown.",
        "columns": {
            "day": "integer 1..28",
            "critical_capacity": "integer >= 0, max units/day from Supplier_Critical",
            "alternate_capacity": "integer >= 0, max units/day from Supplier_Alt",
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
    "production_orders.csv": {
        "description": (
            "Simulated daily supply-chain outcomes. "
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


def generate(seed: int, output_dir: Path):
    rng = np.random.default_rng(seed)  # use the new Generator API for reproducibility

    days = 28

    # ------------------------------------------------------------------ #
    # Entities
    # ------------------------------------------------------------------ #
    critical_supplier = "Supplier_Critical"
    alternate_supplier = "Supplier_Alt"
    distribution_centers = ["DC_1", "DC_2", "DC_3"]

    # ------------------------------------------------------------------ #
    # Static planning parameters
    # ------------------------------------------------------------------ #
    daily_demand_mean = 100       # units / day / DC
    daily_demand_std = 15
    plant_capacity = 250          # max finished units / day
    initial_finished_inventory = 0
    initial_component_inventory = 0

    lead_time_critical = 2        # days
    lead_time_alt = 3             # days
    critical_capacity = 200       # max units / day (when operational)
    alt_capacity = 80             # max units / day
    transport_cost_per_unit = 0.50  # USD / unit shipped
    expediting_surcharge = 50.00    # USD / expedited shipment
    service_fill_rate_target = 0.95  # fraction of demand to be met on time
    shutdown_start = 10           # first day of shutdown (1-indexed)
    shutdown_length = 7           # consecutive days unavailable

    # ------------------------------------------------------------------ #
    # PLANNING INPUT 1 – demand.csv
    # ------------------------------------------------------------------ #
    demand_records = []
    for day in range(1, days + 1):
        for dc in distribution_centers:
            demand = max(0, int(rng.normal(daily_demand_mean, daily_demand_std)))
            demand_records.append({"day": day, "distribution_center": dc, "demand": demand})
    demand_df = pd.DataFrame(demand_records)

    # ------------------------------------------------------------------ #
    # PLANNING INPUT 2 – supplier_capacity.csv
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

    supplier_capacity_df = pd.DataFrame({
        "day": range(1, days + 1),
        "critical_capacity": critical_daily_cap,
        "alternate_capacity": alt_daily_cap,
    })

    # ------------------------------------------------------------------ #
    # PLANNING INPUT 3 – network_params.csv (static, no randomness)
    # ------------------------------------------------------------------ #
    network_params = [
        ("plant_production_capacity_per_day", plant_capacity, "units/day", "Max finished units Plant_A can make per day"),
        ("initial_finished_inventory",        initial_finished_inventory, "units", "Finished-product stock at day 0"),
        ("initial_component_inventory",       initial_component_inventory, "units", "Component stock at day 0"),
        ("lead_time_critical_days",           lead_time_critical, "days", "Lead time from Supplier_Critical to Plant_A"),
        ("lead_time_alt_days",                lead_time_alt, "days", "Lead time from Supplier_Alt to Plant_A"),
        ("critical_supplier_capacity",        critical_capacity, "units/day", "Max daily capacity of Supplier_Critical (when operational)"),
        ("alternate_supplier_capacity",       alt_capacity, "units/day", "Max daily capacity of Supplier_Alt"),
        ("transport_cost_per_unit",           transport_cost_per_unit, "USD/unit", "Standard transport cost per component unit shipped"),
        ("expediting_surcharge",              expediting_surcharge, "USD/shipment", "Extra cost charged when a shipment is expedited"),
        ("service_fill_rate_target",          service_fill_rate_target, "fraction", "Target fraction of daily demand fulfilled on time"),
        ("shutdown_start_day",                shutdown_start, "day (1-indexed)", "First day Supplier_Critical is unavailable"),
        ("shutdown_length_days",              shutdown_length, "days", "Number of consecutive days the critical supplier is shut down"),
        ("demand_mean_per_dc_per_day",        daily_demand_mean, "units/day", "Mean demand per DC per day (Normal distribution)"),
        ("demand_std_per_dc_per_day",         daily_demand_std, "units/day", "Std-dev of demand per DC per day"),
    ]
    network_params_df = pd.DataFrame(network_params, columns=["parameter", "value", "unit", "notes"])

    # ------------------------------------------------------------------ #
    # SIMULATED OUTCOME – production_orders.csv
    # ------------------------------------------------------------------ #
    # Keep planning inputs (demand, capacities) separate from outcomes.
    # The simulation rows below represent what *actually happened* under a
    # simple reactive ordering policy.  Agents / optimisers should use only
    # demand.csv, supplier_capacity.csv, and network_params.csv as inputs.
    # ------------------------------------------------------------------ #
    records = []
    inventory = initial_finished_inventory
    backlog = 0
    component_inventory = initial_component_inventory
    in_transit = []  # {arrival_day, qty, supplier}

    for day in range(1, days + 1):
        # --- Arrivals ---
        for shipment in in_transit:
            if shipment["arrival_day"] == day:
                component_inventory += shipment["qty"]
        in_transit = [s for s in in_transit if s["arrival_day"] != day]

        # --- Production ---
        total_demand_today = int(demand_df[demand_df["day"] == day]["demand"].sum())
        target_produce = total_demand_today + backlog
        max_producible = min(plant_capacity, component_inventory)
        produced = min(max_producible, target_produce)
        component_inventory -= produced
        # --- Demand fulfilment ---
        available = inventory + produced
        fulfilled = min(available, total_demand_today + backlog)
        # inventory_end = whatever is left after fulfilling all we can
        inventory = available - fulfilled
        backlog = (total_demand_today + backlog) - fulfilled

        # Production record
        records.append({
            "day": day,
            "record_type": "production",
            "produced": produced,
            "inventory_end": inventory,
            "backlog_end": backlog,
            "component_inventory_end": component_inventory,
            "supplier": None,
            "order_qty": None,
            "lead_time": None,
            "expedited": None,
        })

        # --- Ordering policy: replenish component stock ---
        target_stock = int(total_demand_today * 1.2)
        order_qty = max(0, target_stock - component_inventory)
        crit_cap = critical_daily_cap[day - 1]
        alt_cap = alt_daily_cap[day - 1]
        total_cap = crit_cap + alt_cap
        if total_cap > 0 and order_qty > 0:
            crit_order = min(int(order_qty * crit_cap / total_cap), crit_cap)
            alt_order = min(order_qty - crit_order, alt_cap)
        else:
            crit_order = 0
            alt_order = 0

        for supplier, qty, lt in [
            (critical_supplier, crit_order, lead_time_critical),
            (alternate_supplier, alt_order, lead_time_alt),
        ]:
            if qty > 0:
                in_transit.append({"arrival_day": day + lt, "qty": qty, "supplier": supplier})
            records.append({
                "day": day,
                "record_type": "order",
                "produced": None,
                "inventory_end": None,
                "backlog_end": None,
                "component_inventory_end": None,
                "supplier": supplier,
                "order_qty": qty,
                "lead_time": lt,
                "expedited": False,
            })

    production_df = pd.DataFrame(records)

    # ------------------------------------------------------------------ #
    # Write output
    # ------------------------------------------------------------------ #
    os.makedirs(output_dir, exist_ok=True)
    demand_df.to_csv(output_dir / "demand.csv", index=False)
    supplier_capacity_df.to_csv(output_dir / "supplier_capacity.csv", index=False)
    network_params_df.to_csv(output_dir / "network_params.csv", index=False)
    production_df.to_csv(output_dir / "production_orders.csv", index=False)

    meta = {
        "seed": seed,
        "days": days,
        "critical_supplier_shutdown": {
            "start_day": shutdown_start,
            "length": shutdown_length,
        },
        "units": "pieces",
        "schema": SCHEMA,
        "assumptions": [
            "One component required per finished unit (1:1 BOM).",
            "Demand is independently normally distributed across DCs and days.",
            "The ordering policy is reactive (not optimal): order 1.2x today's demand minus current component stock.",
            "Orders are split proportionally to supplier capacity on the day.",
            "Alternate supplier can always be used within its capacity; it is not shut down.",
            "Production and ordering happen within the same day; arrivals happen at the start of each day.",
            "No transit-time variability (lead times are deterministic).",
            "Transport cost is charged per unit ordered; expediting surcharge is per shipment.",
            "Service level is measured as fill-rate (fraction of demand fulfilled on time).",
            "All costs are in USD; all quantities in pieces.",
        ],
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic 28-day supply-chain dataset.")
    parser.add_argument("--output", type=Path, required=True, help="Directory to write CSV/JSON files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    generate(args.seed, args.output)


if __name__ == "__main__":
    main()
