import os
import json
import hashlib
import pandas as pd
from pathlib import Path
import tempfile
import filecmp

from scripts.generate_dataset import generate


def hash_file(path: Path) -> str:
    """Return MD5 hash of a file's contents."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_reproducibility():
    # Generate two datasets with same seed in separate temp dirs
    seed = 12345
    with tempfile.TemporaryDirectory() as dir1, tempfile.TemporaryDirectory() as dir2:
        out1 = Path(dir1)
        out2 = Path(dir2)
        generate(seed, out1)
        generate(seed, out2)
        # Compare hashes of each generated file
        for fname in ["demand.csv", "supplier_capacity.csv", "production_orders.csv", "metadata.json"]:
            assert hash_file(out1 / fname) == hash_file(out2 / fname), f"File {fname} differs between runs"


def test_non_negative_quantities():
    seed = 42
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(seed, out)
        # Load CSVs
        demand = pd.read_csv(out / "demand.csv")
        supplier_cap = pd.read_csv(out / "supplier_capacity.csv")
        production = pd.read_csv(out / "production_orders.csv")
        # Numeric columns should be >= 0
        for df in [demand, supplier_cap, production]:
            numeric_cols = df.select_dtypes(include="number").columns
            assert (df[numeric_cols] >= 0).all().all()


def test_missing_data():
    seed = 7
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(seed, out)
        for fname in ["demand.csv", "supplier_capacity.csv", "production_orders.csv", "metadata.json"]:
            path = out / fname
            # Ensure file exists and is not empty
            assert path.is_file()
            assert path.stat().st_size > 0
            # Simple check: no completely empty lines in CSVs
            if fname.endswith('.csv'):
                with open(path) as f:
                    for line in f:
                        assert line.strip() != "", f"Empty line in {fname}"


def test_shutdown_duration():
    seed = 99
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        generate(seed, out)
        meta_path = out / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        shutdown = meta.get("critical_supplier_shutdown", {})
        assert shutdown.get("length") == 7, "Critical supplier shutdown length should be 7 days"
        # Verify that during shutdown days, critical capacity is zero
        supplier_cap = pd.read_csv(out / "supplier_capacity.csv")
        start_day = shutdown.get("start_day")
        length = shutdown.get("length")
        shutdown_days = list(range(start_day, start_day + length))
        for day in shutdown_days:
            cap = supplier_cap[supplier_cap["day"] == day]["critical_capacity"].iloc[0]
            assert cap == 0, f"Critical capacity on day {day} should be 0"

