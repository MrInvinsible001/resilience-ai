"""
test_invariants.py – cross-file consistency, inventory-flow equation,
supplier capacity constraints, shutdown enforcement, and experimental design invariants.
"""
import json
import pandas as pd
from pathlib import Path
import pytest
import tempfile

from scripts.generate_dataset import generate

SEED = 42


def _load(out: Path):
    demand = pd.read_csv(out / "demand.csv")
    sup_cap = pd.read_csv(out / "supplier_capacity.csv")
    sup_cap_baseline = pd.read_csv(out / "supplier_capacity_baseline.csv")
    net_par = pd.read_csv(out / "network_params.csv")
    prod = pd.read_csv(out / "production_orders.csv")
    with open(out / "metadata.json") as f:
        meta = json.load(f)
    with open(out / "initial_conditions.json") as f:
        init_cond = json.load(f)
    with open(out / "disruption_events.json") as f:
        disruptions = json.load(f)
    return demand, sup_cap, sup_cap_baseline, net_par, prod, meta, init_cond, disruptions


# ------------------------------------------------------------------
# 1. Cross-file row counts
# ------------------------------------------------------------------
def test_row_counts():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        demand, sup_cap, sup_cap_baseline, net_par, prod, meta, _, _ = _load(out)
        days = meta["days"]
        dcs = 3

        assert len(demand) == days * dcs, (
            f"demand.csv: expected {days*dcs} rows, got {len(demand)}"
        )
        assert len(sup_cap) == days, (
            f"supplier_capacity.csv: expected {days} rows, got {len(sup_cap)}"
        )
        assert len(sup_cap_baseline) == days, (
            f"supplier_capacity_baseline.csv: expected {days} rows, got {len(sup_cap_baseline)}"
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
# 2. Inventory-flow global equation (accounting for initial inventory)
#    total_produced + final_backlog == total_demand + final_inventory - initial_inventory
# ------------------------------------------------------------------
def test_inventory_flow_equation():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        demand, sup_cap, _, net_par, prod, meta, init_cond, _ = _load(out)

        p_rows = prod[prod["record_type"] == "production"]
        total_demand = demand["demand"].sum()
        total_produced = p_rows["produced"].sum()
        final_backlog = p_rows["backlog_end"].iloc[-1]
        final_inventory = p_rows["inventory_end"].iloc[-1]
        initial_inventory = init_cond.get("finished_inventory", 0)

        lhs = total_produced + final_backlog
        rhs = total_demand + final_inventory - initial_inventory
        assert lhs == rhs, (
            f"Inventory-flow violated: produced({total_produced}) + backlog({final_backlog}) "
            f"= {lhs} != demand({total_demand}) + inventory({final_inventory}) - I0({initial_inventory}) = {rhs}"
        )
        assert final_inventory >= 0, "Final finished-product inventory must be >= 0"
        assert final_backlog >= 0, "Final backlog must be >= 0"


# ------------------------------------------------------------------
# 2b. Component-inventory global flow equation on exported CSVs
#     total_arrivals == total_produced + final_components - initial_components
# ------------------------------------------------------------------
def test_component_inventory_flow_equation():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        _, _, _, _, prod, _, init_cond, _ = _load(out)

        c0 = init_cond["component_inventory"]
        init_arrivals = sum(s["quantity"] for s in init_cond["in_transit"] if s["arrival_day"] <= 28)

        p_rows = prod[prod["record_type"] == "production"]
        o_rows = prod[prod["record_type"] == "order"]

        total_produced = p_rows["produced"].sum()
        c28 = p_rows["component_inventory_end"].iloc[-1]

        order_arrivals = sum(
            row["order_qty"]
            for _, row in o_rows.iterrows()
            if row["day"] + row["lead_time"] <= 28
        )

        total_arrivals = init_arrivals + order_arrivals
        lhs = total_arrivals
        rhs = total_produced + c28 - c0

        assert lhs == rhs, (
            f"Component flow violated: arrivals({total_arrivals}) != "
            f"produced({total_produced}) + C28({c28}) - C0({c0}) = {rhs}"
        )
        assert c28 >= 0, "Final component inventory must be >= 0"


# ------------------------------------------------------------------
# 3. Component inventory never goes negative
# ------------------------------------------------------------------
def test_component_inventory_non_negative():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        _, _, _, _, prod, meta, _, _ = _load(out)
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
        demand, sup_cap, _, net_par, prod, meta, _, _ = _load(out)
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
        demand, sup_cap, _, net_par, prod, meta, _, _ = _load(out)
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
            (o_rows["supplier"] == "Supplier_Critical")
            & (o_rows["day"] >= start)
            & (o_rows["day"] < start + length)
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
        _, _, _, net_par, _, _, _, _ = _load(out)
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
        demand, sup_cap, sup_cap_baseline, net_par, prod, meta, _, _ = _load(out)
        days = meta["days"]
        expected_days = set(range(1, days + 1))

        assert set(demand["day"].unique()) == expected_days, "demand.csv missing some days"
        assert set(sup_cap["day"].unique()) == expected_days, "supplier_capacity.csv missing some days"
        assert set(sup_cap_baseline["day"].unique()) == expected_days, "supplier_capacity_baseline.csv missing some days"
        assert set(prod["day"].unique()) == expected_days, "production_orders.csv missing some days"


# ------------------------------------------------------------------
# 8. Capacity Feasibility Tests (CF1 - CF4)
# ------------------------------------------------------------------
def test_capacity_exceeds_mean_demand():
    """CF1: Plant capacity (300) >= total mean demand (240)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        _, _, _, net_par, _, _, _, _ = _load(out)
        params = dict(zip(net_par["parameter"], net_par["value"]))
        plant_cap = params["plant_production_capacity_per_day"]
        mean_demand = params["demand_mean_per_dc_per_day"] * 3
        assert plant_cap >= mean_demand, f"Plant capacity {plant_cap} < mean demand {mean_demand}"


def test_combined_suppliers_cover_plant():
    """CF2: Combined normal supplier capacity (220 + 100 = 320) >= plant capacity (300)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        _, _, _, net_par, _, _, _, _ = _load(out)
        params = dict(zip(net_par["parameter"], net_par["value"]))
        crit_cap = params["critical_supplier_capacity"]
        alt_cap = params["alternate_supplier_capacity"]
        plant_cap = params["plant_production_capacity_per_day"]
        assert crit_cap + alt_cap >= plant_cap, (
            f"Combined supplier capacity {crit_cap + alt_cap} < plant capacity {plant_cap}"
        )


def test_alternate_alone_insufficient():
    """CF3: Alternate supplier capacity (100) < total mean demand (240)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        _, _, _, net_par, _, _, _, _ = _load(out)
        params = dict(zip(net_par["parameter"], net_par["value"]))
        alt_cap = params["alternate_supplier_capacity"]
        mean_demand = params["demand_mean_per_dc_per_day"] * 3
        assert alt_cap < mean_demand, (
            f"Alternate capacity {alt_cap} should be strictly less than mean demand {mean_demand}"
        )


def test_alternate_cannot_fill_plant():
    """CF4: Alternate supplier capacity (100) < plant capacity (300)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        _, _, _, net_par, _, _, _, _ = _load(out)
        params = dict(zip(net_par["parameter"], net_par["value"]))
        alt_cap = params["alternate_supplier_capacity"]
        plant_cap = params["plant_production_capacity_per_day"]
        assert alt_cap < plant_cap, (
            f"Alternate capacity {alt_cap} should be strictly less than plant capacity {plant_cap}"
        )


# ------------------------------------------------------------------
# 9. Burn-in Initialization Tests (IN1 - IN5)
# ------------------------------------------------------------------
def test_initial_backlog_is_zero():
    """IN1: initial_conditions['backlog'] == 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        _, _, _, _, _, _, init_cond, _ = _load(out)
        assert init_cond["backlog"] == 0, f"Initial backlog must be 0, got {init_cond['backlog']}"


def test_warmstart_inventory_positive():
    """IN2: initial_conditions['finished_inventory'] > 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        _, _, _, _, _, _, init_cond, _ = _load(out)
        assert init_cond["finished_inventory"] > 0, (
            f"Initial finished inventory must be > 0, got {init_cond['finished_inventory']}"
        )


def test_warmstart_components_nonneg():
    """IN3: initial_conditions['component_inventory'] >= 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        _, _, _, _, _, _, init_cond, _ = _load(out)
        assert init_cond["component_inventory"] >= 0, (
            f"Initial component inventory must be >= 0, got {init_cond['component_inventory']}"
        )


def test_warmstart_pipeline_nonempty():
    """IN4: len(initial_conditions['in_transit']) >= 1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        _, _, _, _, _, _, init_cond, _ = _load(out)
        assert len(init_cond["in_transit"]) >= 1, "Initial pipeline must contain active shipments"


def test_static_capacity_guard_fires_on_infeasible_plant():
    """IN5a: generate() with plant capacity below mean demand raises static feasibility error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        with pytest.raises(ValueError, match="Infeasible network parameters"):
            generate(SEED, out, plant_capacity=50)


def test_burnin_runtime_backlog_guard_fires():
    """IN5b: generate() with tight plant capacity (241) fails burn-in runtime backlog guard."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        with pytest.raises(ValueError, match="Burn-in backlog guard failed"):
            generate(SEED, out, plant_capacity=241)


# ------------------------------------------------------------------
# 10. Shutdown Boundary and Baseline Tests (ST3 - ST5)
# ------------------------------------------------------------------
def test_critical_capacity_normal_outside_shutdown():
    """ST3: Critical capacity == 220 on days 1..9 and 17..28."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        _, sup_cap, _, _, _, _, _, _ = _load(out)
        non_shutdown_days = [d for d in range(1, 29) if d < 10 or d > 16]
        for d in non_shutdown_days:
            cap = sup_cap[sup_cap["day"] == d]["critical_capacity"].iloc[0]
            assert cap == 220, f"Day {d} critical capacity should be 220, got {cap}"


def test_scenario_b_baseline_hides_shutdown():
    """ST4: supplier_capacity_baseline.csv has 220 critical capacity on all days (no zeros)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        _, _, sup_cap_baseline, _, _, _, _, _ = _load(out)
        assert (sup_cap_baseline["critical_capacity"] == 220).all(), (
            "Baseline capacity must show 220 on all days without shutdown"
        )
        assert (sup_cap_baseline["critical_capacity"] > 0).all(), "Baseline must not contain zeros"


def test_disruption_events_correct_window():
    """ST5: disruption_events.json records shutdown starting at day 10 with length 7."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        _, _, _, _, _, _, _, disruptions = _load(out)
        events = disruptions["disruptions"]
        assert len(events) >= 1
        ev = events[0]
        assert ev["supplier_id"] == "Supplier_Critical"
        assert ev["start_day"] == 10
        assert ev["length_days"] == 7
        assert ev["capacity_during_disruption"] == 0


# ------------------------------------------------------------------
# 11. Accounting & Production Target Tests (AC5, AC7)
# ------------------------------------------------------------------
def test_production_within_plant_capacity():
    """AC5: All produced <= plant_capacity (300)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        _, _, _, _, prod, _, _, _ = _load(out)
        p_rows = prod[prod["record_type"] == "production"]
        assert (p_rows["produced"] <= 300).all(), "Production exceeded plant capacity 300"


def test_no_overproduction():
    """AC7: Production target respects existing finished goods."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        demand, _, _, _, prod, _, init_cond, _ = _load(out)
        p_rows = prod[prod["record_type"] == "production"].reset_index(drop=True)
        # On day 1: demand is ~235, initial inventory is ~100.
        # Target = max(0, demand - initial_inventory). Produced should be <= target.
        d1_demand = demand[demand["day"] == 1]["demand"].sum()
        i0 = init_cond["finished_inventory"]
        p1 = p_rows.loc[0, "produced"]
        expected_target = max(0, d1_demand - i0)
        assert p1 <= expected_target, f"Day 1 produced {p1} exceeds target {expected_target}"


# ------------------------------------------------------------------
# 12. Disruption Impact Tests (SF1 - SF3)
# ------------------------------------------------------------------
def test_no_pre_shutdown_backlog():
    """SF1: Day 9 backlog == 0 under baseline policy with warmstart."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        _, _, _, _, prod, _, _, _ = _load(out)
        p_rows = prod[prod["record_type"] == "production"].reset_index(drop=True)
        day_9_backlog = p_rows.loc[8, "backlog_end"]
        assert day_9_backlog == 0, f"Day 9 backlog should be 0 before shutdown, got {day_9_backlog}"


def test_shutdown_causes_backlog():
    """SF2: Day 16 backlog > 0 due to 7-day critical shutdown."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        _, _, _, _, prod, _, _, _ = _load(out)
        p_rows = prod[prod["record_type"] == "production"].reset_index(drop=True)
        day_16_backlog = p_rows.loc[15, "backlog_end"]
        assert day_16_backlog > 0, f"Day 16 backlog should be > 0 at end of shutdown, got {day_16_backlog}"


def test_recovery_possible():
    """SF3: Day 28 backlog < Day 16 backlog (recovery occurs post-shutdown)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        _, _, _, _, prod, _, _, _ = _load(out)
        p_rows = prod[prod["record_type"] == "production"].reset_index(drop=True)
        day_16_backlog = p_rows.loc[15, "backlog_end"]
        day_28_backlog = p_rows.loc[27, "backlog_end"]
        assert day_28_backlog < day_16_backlog, (
            f"Day 28 backlog ({day_28_backlog}) should be lower than Day 16 backlog ({day_16_backlog})"
        )
