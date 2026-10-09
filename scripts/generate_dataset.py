import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd


def generate(seed: int, output_dir: Path):
    np.random.seed(seed)
    days = 28
    # Entities
    plant = "Plant_A"
    product = "Finished_Product"
    component = "Component_X"
    critical_supplier = "Supplier_Critical"
    alternate_supplier = "Supplier_Alt"
    distribution_centers = ["DC_1", "DC_2", "DC_3"]

    # Parameters (units per day, days, costs)
    daily_demand_mean = 100
    daily_demand_std = 15
    plant_capacity = 250  # max production per day
    lead_time_critical = 2
    lead_time_alt = 3
    critical_capacity = 200  # units per day from critical supplier
    alt_capacity = 80  # units per day from alternate supplier
    transport_cost_per_unit = 0.5
    expediting_surcharge = 50  # per shipment when expedited
    shutdown_start = 10  # day index (0-based) when critical supplier shuts down
    shutdown_length = 7

    # Generate daily demand per distribution centre
    demand_records = []
    for day in range(1, days + 1):
        for dc in distribution_centers:
            demand = max(0, int(np.random.normal(daily_demand_mean, daily_demand_std)))
            demand_records.append({"day": day, "distribution_center": dc, "demand": demand})
    demand_df = pd.DataFrame(demand_records)

    # Supplier availability (capacity per day)
    critical_daily_capacity = []
    alt_daily_capacity = []
    for day in range(1, days + 1):
        if shutdown_start <= day < shutdown_start + shutdown_length:
            crit_cap = 0
        else:
            crit_cap = critical_capacity
        critical_daily_capacity.append(crit_cap)
        alt_daily_capacity.append(alt_capacity)
    supplier_capacity_df = pd.DataFrame({
        "day": range(1, days + 1),
        "critical_capacity": critical_daily_capacity,
        "alternate_capacity": alt_daily_capacity,
    })

    # Simulate ordering policy: order enough to meet expected demand + safety stock
    # For simplicity we order the daily demand sum divided equally between suppliers respecting capacities.
    orders = []
    inventory = 0
    backlog = 0
    component_inventory = 0
    # Track in‑transit shipments
    in_transit = []  # list of dicts with arrival_day, qty, supplier
    for day in range(1, days + 1):
        # arrivals
        arrivals = [t for t in in_transit if t["arrival_day"] == day]
        for arr in arrivals:
            component_inventory += arr["qty"]
        in_transit = [t for t in in_transit if t["arrival_day"] != day]

        # demand for the day
        day_demand = demand_df[demand_df["day"] == day]["demand"].sum()
        # production limited by plant capacity and component availability
        max_producible = min(plant_capacity, component_inventory)
        produced = min(max_producible, day_demand + backlog)  # try to satisfy demand + backlog
        component_inventory -= produced
        inventory += produced
        # demand fulfillment
        fulfilled = min(day_demand + backlog, produced)
        backlog = (day_demand + backlog) - fulfilled
        # record production
        orders.append({
            "day": day,
            "produced": produced,
            "inventory_end": inventory,
            "backlog_end": backlog,
            "component_inventory_end": component_inventory,
        })
        # decide next day orders from suppliers based on expected demand
        # simple heuristic: order (day_demand * 1.2) - component_inventory
        target_stock = int(day_demand * 1.2)
        order_qty = max(0, target_stock - component_inventory)
        # split order between available suppliers proportionally to their capacities
        crit_cap = critical_daily_capacity[day - 1]
        alt_cap = alt_daily_capacity[day - 1]
        total_cap = crit_cap + alt_cap
        if total_cap > 0 and order_qty > 0:
            crit_order = int(order_qty * (crit_cap / total_cap))
            alt_order = order_qty - crit_order
        else:
            crit_order = 0
            alt_order = 0
        # schedule arrivals based on lead times
        if crit_order > 0:
            arrival_day = day + lead_time_critical
            in_transit.append({"arrival_day": arrival_day, "qty": crit_order, "supplier": critical_supplier})
        if alt_order > 0:
            arrival_day = day + lead_time_alt
            in_transit.append({"arrival_day": arrival_day, "qty": alt_order, "supplier": alternate_supplier})
        # record orders
        orders.append({
            "day": day,
            "supplier": critical_supplier,
            "order_qty": crit_order,
            "lead_time": lead_time_critical,
            "expedited": False,
        })
        orders.append({
            "day": day,
            "supplier": alternate_supplier,
            "order_qty": alt_order,
            "lead_time": lead_time_alt,
            "expedited": False,
        })

    # Build DataFrames
    production_df = pd.DataFrame(orders)
    # Replace missing values (NaN) with 0 to satisfy non‑negative checks
    production_df = production_df.fillna(0)
    # Save files
    os.makedirs(output_dir, exist_ok=True)
    demand_df.to_csv(output_dir / "demand.csv", index=False)
    supplier_capacity_df.to_csv(output_dir / "supplier_capacity.csv", index=False)
    production_df.to_csv(output_dir / "production_orders.csv", index=False)
    # Metadata
    meta = {
        "seed": seed,
        "days": days,
        "critical_supplier_shutdown": {"start_day": shutdown_start, "length": shutdown_length},
        "units": "units (e.g., pieces)",
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic 28‑day supply‑chain dataset.")
    parser.add_argument("--output", type=Path, required=True, help="Directory to write CSV/JSON files.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    generate(args.seed, args.output)


if __name__ == "__main__":
    main()

