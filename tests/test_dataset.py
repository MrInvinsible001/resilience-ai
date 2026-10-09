"""
test_dataset.py – basic tests: reproducibility, non-negative quantities,
missing data, and shutdown duration.
"""
import hashlib
import json
import pandas as pd
from pathlib import Path
import tempfile

from scripts.generate_dataset import generate

SEED = 42
EXPECTED_FILES = [
    "demand.csv",
    "supplier_capacity.csv",
    "supplier_capacity_baseline.csv",
    "disruption_events.json",
    "network_params.csv",
    "initial_conditions.json",
    "production_orders.csv",
    "metadata.json",
]


def _generate_pair(seed=SEED):
    """Return two independently-generated Path objects for the same seed."""
    d1 = tempfile.mkdtemp()
    d2 = tempfile.mkdtemp()
    generate(seed, Path(d1))
    generate(seed, Path(d2))
    return Path(d1), Path(d2)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------
# 1. Reproducibility
# ------------------------------------------------------------------
def test_reproducibility():
    out1, out2 = _generate_pair()
    for fname in EXPECTED_FILES:
        assert _md5(out1 / fname) == _md5(out2 / fname), (
            f"{fname} differs between two runs with the same seed"
        )


# ------------------------------------------------------------------
# 2. All expected files exist and are non-empty
# ------------------------------------------------------------------
def test_expected_files_exist():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        for fname in EXPECTED_FILES:
            p = out / fname
            assert p.is_file(), f"{fname} is missing"
            assert p.stat().st_size > 0, f"{fname} is empty"


# ------------------------------------------------------------------
# 3. Non-negative quantities in every CSV
# ------------------------------------------------------------------
def test_non_negative_quantities():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        for fname in ["demand.csv", "supplier_capacity.csv", "supplier_capacity_baseline.csv"]:
            df = pd.read_csv(out / fname)
            nums = df.select_dtypes(include="number")
            assert (nums >= 0).all().all(), f"Negative values found in {fname}"

        # For production_orders: check numeric columns per record_type
        prod = pd.read_csv(out / "production_orders.csv")
        p_rows = prod[prod["record_type"] == "production"]
        for col in ["produced", "inventory_end", "backlog_end", "component_inventory_end"]:
            assert (p_rows[col] >= 0).all(), f"Negative {col} in production rows"
        o_rows = prod[prod["record_type"] == "order"]
        assert (o_rows["order_qty"] >= 0).all(), "Negative order_qty in order rows"
        assert (o_rows["lead_time"] >= 0).all(), "Negative lead_time in order rows"


# ------------------------------------------------------------------
# 4. Shutdown duration is exactly 7 days
# ------------------------------------------------------------------
def test_shutdown_duration():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(SEED, out)
        with open(out / "metadata.json") as f:
            meta = json.load(f)
        shutdown = meta["critical_supplier_shutdown"]
        assert shutdown["length"] == 7, "Shutdown length must be 7"

        cap = pd.read_csv(out / "supplier_capacity.csv")
        start = shutdown["start_day"]
        length = shutdown["length"]
        shutdown_days = range(start, start + length)
        for d in shutdown_days:
            row_cap = cap[cap["day"] == d]["critical_capacity"].iloc[0]
            assert row_cap == 0, (
                f"Critical capacity should be 0 on shutdown day {d}, got {row_cap}"
            )
