"""
test_invariants.py – cross-file consistency, inventory-flow equation,
supplier capacity constraints, and shutdown enforcement.
"""
import json
import pandas as pd
from pathlib import Path
import tempfile

from scripts.generate_dataset import generate

SEED = 42


def _load(out: Path):
    demand = pd.read_csv(out / "demand.csv")
    sup_cap = pd.read_csv(out / "supplier_capacity.csv")
    net_par = pd.read_csv(out / "network_params.csv")
    prod = pd.read_csv(out / "production_orders.csv")
    with open(out / "metadata.json") as f:
        meta = json.load(f)
    return demand, sup_cap, net_par, prod, meta


# ------------------------------------------------------------------
# 1. Cross-file row counts
# ------------------------------------------------------------------
def test_row_counts():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        demand, sup_cap, net_par, prod, meta = _load(out)
        days = meta["days"]
        dcs = 3

        assert len(demand) == days * dcs, (
            f"demand.csv: expected {days*dcs} rows, got {len(demand)}"
        )
        assert len(sup_cap) == days, (
            f"supplier_capacity.csv: expected {days} rows, got {len(sup_cap)}"
        )

        p_rows = prod[prod["record_type"] == "production"]
        o_rows = prod[prod["record_type"] == "order"]
        assert len(p_rows) == days, (
            f"Expected {days} production rows, got {len(p_rows)}"
        )
        # 2 order rows per day (one per supplier)
        assert len(o_rows) == days * 2, (
            f"Expected {days*2} order rows, got {len(o_rows)}"
        )


# ------------------------------------------------------------------
# 2. Inventory-flow global equation
#    total_produced + final_backlog == total_demand + final_inventory
# ------------------------------------------------------------------
def test_inventory_flow_equation():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        demand, sup_cap, net_par, prod, meta = _load(out)

        p_rows = prod[prod["record_type"] == "production"]
        total_demand = demand["demand"].sum()
        total_produced = p_rows["produced"].sum()
        final_backlog = p_rows["backlog_end"].iloc[-1]
        final_inventory = p_rows["inventory_end"].iloc[-1]

        lhs = total_produced + final_backlog
        rhs = total_demand + final_inventory
        assert lhs == rhs, (
            f"Inventory-flow violated: produced({total_produced}) + backlog({final_backlog}) "
            f"= {lhs} != demand({total_demand}) + inventory({final_inventory}) = {rhs}"
        )
        assert final_inventory >= 0, "Final finished-product inventory must be >= 0"
        assert final_backlog >= 0, "Final backlog must be >= 0"


# ------------------------------------------------------------------
# 3. Component inventory never goes negative
# ------------------------------------------------------------------
def test_component_inventory_non_negative():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        _, _, _, prod, meta = _load(out)
        p_rows = prod[prod["record_type"] == "production"]
        for _, row in p_rows.iterrows():
            assert row["component_inventory_end"] >= 0, (
                f"Component inventory negative on day {int(row['day'])}: {row['component_inventory_end']}"
            )


# ------------------------------------------------------------------
# 4. Supplier capacity constraints respected in every order row
# ------------------------------------------------------------------
def test_supplier_capacity_constraints():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        demand, sup_cap, net_par, prod, meta = _load(out)
        days = meta["days"]
        o_rows = prod[prod["record_type"] == "order"]
        for day in range(1, days + 1):
            row_cap = sup_cap[sup_cap["day"] == day].iloc[0]
            crit_cap = row_cap["critical_capacity"]
            alt_cap = row_cap["alternate_capacity"]
            day_orders = o_rows[o_rows["day"] == day]
            crit_qty = day_orders[day_orders["supplier"] == "Supplier_Critical"]["order_qty"].sum()
            alt_qty = day_orders[day_orders["supplier"] == "Supplier_Alt"]["order_qty"].sum()
            assert crit_qty <= crit_cap, (
                f"Day {day}: critical order {crit_qty} exceeds capacity {crit_cap}"
            )
            assert alt_qty <= alt_cap, (
                f"Day {day}: alternate order {alt_qty} exceeds capacity {alt_cap}"
            )


# ------------------------------------------------------------------
# 5. Exactly 7 consecutive shutdown days; orders = 0 on those days
# ------------------------------------------------------------------
def test_shutdown_exactly_7_consecutive_days():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        demand, sup_cap, net_par, prod, meta = _load(out)
        shutdown = meta["critical_supplier_shutdown"]
        start = shutdown["start_day"]
        length = shutdown["length"]

        assert length == 7, f"Shutdown length must be 7, got {length}"

        # All 7 days consecutive – capacity is zero
        zero_cap_days = sup_cap[sup_cap["critical_capacity"] == 0]["day"].tolist()
        assert sorted(zero_cap_days) == list(range(start, start + length)), (
            f"Shutdown days in supplier_capacity.csv not exactly {list(range(start, start + length))}: "
            f"got {sorted(zero_cap_days)}"
        )

        # No critical orders placed during shutdown
        o_rows = prod[prod["record_type"] == "order"]
        crit_orders = o_rows[
            (o_rows["supplier"] == "Supplier_Critical") &
            (o_rows["day"] >= start) &
            (o_rows["day"] < start + length)
        ]
        for _, row in crit_orders.iterrows():
            assert row["order_qty"] == 0, (
                f"Critical order on shutdown day {int(row['day'])}: qty = {row['order_qty']}"
            )


# ------------------------------------------------------------------
# 6. network_params.csv contains all required planning inputs
# ------------------------------------------------------------------
def test_network_params_completeness():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        _, _, net_par, _, _ = _load(out)
        required = {
            "plant_production_capacity_per_day",
            "initial_finished_inventory",
            "initial_component_inventory",
            "lead_time_critical_days",
            "lead_time_alt_days",
            "critical_supplier_capacity",
            "alternate_supplier_capacity",
            "transport_cost_per_unit",
            "expediting_surcharge",
            "service_fill_rate_target",
            "shutdown_start_day",
            "shutdown_length_days",
            "demand_mean_per_dc_per_day",
            "demand_std_per_dc_per_day",
        }
        present = set(net_par["parameter"].tolist())
        missing = required - present
        assert not missing, f"Missing required planning parameters: {missing}"


# ------------------------------------------------------------------
# 7. Cross-file: demand days match supplier-capacity days
# ------------------------------------------------------------------
def test_day_index_consistency():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        demand, sup_cap, _, prod, meta = _load(out)
        days = meta["days"]
        expected_days = set(range(1, days + 1))

        assert set(demand["day"].unique()) == expected_days, "demand.csv missing some days"
        assert set(sup_cap["day"].unique()) == expected_days, "supplier_capacity.csv missing some days"
        assert set(prod["day"].unique()) == expected_days, "production_orders.csv missing some days"
