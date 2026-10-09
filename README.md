# ResilienceAI

A lightweight foundation for generating synthetic supply‑chain data for resilience research.

## Overview
This repository provides a reproducible data generator that creates a 28‑day scenario with:
- One manufacturing plant producing a finished product from a single component.
- One critical supplier (primary) and one alternate supplier for the component.
- Three distribution centres serving end‑customer demand.
- Daily demand, inventory levels, capacities, lead‑times, transport and expediting costs.
- A simulated seven‑day shutdown of the critical supplier.

The generated CSV files are placed under `data/` and describe daily inventory flows, orders, shipments, and costs.

## Repository Structure
```
📦
├─ 📂 data/                 # Generated CSV files (not committed)
├─ 📂 scripts/              # Data generation script
│   └─ generate_dataset.py
├─ 📂 tests/                # Basic tests for reproducibility & validity
│   └─ test_dataset.py
├─ .gitignore
├─ README.md
└─ requirements.txt
```

## Assumptions & Units
| Parameter | Unit |
|-----------|------|
| Demand, Production, Inventory | units (e.g., pieces) |
| Capacities | units per day |
| Lead time | days |
| Costs | USD per unit (transport) or per shipment (expediting) |

- All quantities are integer values for simplicity.
- Lead times are whole days.
- Transport costs are per unit shipped; expediting costs are a fixed surcharge per expedited shipment.
- The alternate supplier can satisfy demand but at higher cost and lower capacity.

## Generating the Dataset
```bash
python scripts/generate_dataset.py --output data/ --seed 42
```
The script is deterministic when the same seed is supplied.

## Testing
Run the tests with:
```bash
pytest -q
```
They check that:
- The generated data is identical across runs with the same seed.
- Quantities are non‑negative and respect capacities.
- The shutdown period for the critical supplier is exactly seven days.

## License
This project is MIT licensed – feel free to adapt and extend.

