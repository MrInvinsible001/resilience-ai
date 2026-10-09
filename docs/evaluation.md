# Evaluation Runner

The reusable evaluation API is `resilience_ai.evaluation.run_benchmark`.
It accepts a mapping of strategy names to zero-argument factories, a
`SimulationConfig`, scenario modes, and seeds:

```python
from resilience_ai.evaluation import run_benchmark
from resilience_ai.simulator import BaselineReactiveStrategy, SimulationConfig

report = run_benchmark(
    {"baseline": BaselineReactiveStrategy},
    SimulationConfig(),
    seeds=(42,),
)
summary = report.summary
daily = report.daily_trajectories
```

The summary has one row per strategy, scenario, and seed. The daily table has
one row per strategy, scenario, seed, and evaluation day. Both tables have
stable column names and are pandas DataFrames.

## CLI

Run the baseline strategy under both canonical scenarios:

```bash
python scripts/run_evaluation.py --seed 42 --output results/evaluation
```

The command writes `summary_metrics.csv`, `daily_trajectories.csv`, and
`aggregate_metrics.csv` under the output directory. It never overwrites files
under `data/`. The default when no seed option is supplied is equivalent to
`--seed 42`.

Run multiple seeds with:

```bash
python scripts/run_evaluation.py --seeds 42 43 44 --output results/evaluation
```

`--seed` and `--seeds` are mutually exclusive.

The three output files are:

- `summary_metrics.csv`: one row per strategy, scenario, and seed.
- `daily_trajectories.csv`: one row per strategy, scenario, seed, and day.
- `aggregate_metrics.csv`: one row per strategy and scenario, with
  `trial_count` plus mean and standard deviation columns for the outcome
  metrics listed below. Runtime is intentionally excluded because it is noisy.

## Metric definitions

- **Total realized demand** is the sum of aggregate daily demand.
- **Same-day served** is
  `min(demand, max(0, fulfilled - backlog_start))`, treating old backlog as
  fulfilled before new demand. Same-day fill rate is the sum of same-day served
  divided by total demand. A zero-demand day has a daily rate of `1.0`; a
  zero-demand horizon has a summary rate of `1.0`.
- **Demand satisfied by horizon end** is
  `max(0, total demand - ending backlog)`, with its fraction using total demand
  as the denominator. The warm-start contract begins with zero backlog.
- **Ending backlog**, **peak backlog**, **days with positive backlog**, and
  **cumulative backlog unit-days** use end-of-day backlog.
- **Recovery time** uses day 9 backlog as the baseline. If backlog never rises
  above that baseline on days 10–16, recovery is `0`. Otherwise, recovery is
  the first day from day 17 onward whose backlog is at or below baseline and
  stays at or below baseline through day 28. The reported value is elapsed days
  after day 16; if it never happens, it is missing (`None`/`NaN`).
- **Procurement expenditure** is purchase cost plus transport cost. It excludes
  holding, shortage, backlog, production, and other operating penalties.
- **Order volatility** is the sum of absolute changes in total accepted order
  quantity, starting from zero before day 1.
- **Constraint violations** count all validator violations; capacity violations
  count only `CAPACITY_EXCEEDED`.
- **Accounting invariants** pass only when every simulator invariant is true.
- **Runtime** is wall-clock simulator runtime in milliseconds and is not a
  deterministic performance score.
- **Aggregate recovery time** uses the mean and standard deviation of the
  available (non-missing) recovery times. Missing values are ignored for these
  two statistics; if every trial is missing, both aggregate values are
  missing. `trial_count` still counts all trials in the group. Aggregate
  standard deviations use the population definition (`ddof=0`).

## Fairness and reproducibility

For each seed, the runner creates one `SimulationConfig`, one burn-in
warm-start, and one evaluation demand mapping. Every strategy and scenario
receives the same state and demand mapping for that seed. A fresh strategy
instance is created from its factory for every trial. Strategies continue to
receive only the observations allowed by the simulator, including surprise
shutdown disclosure rules and no future realized demand.
