"""Run the baseline strategy across both canonical shutdown scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from resilience_ai.contracts import ScenarioMode
from resilience_ai.evaluation import run_benchmark
from resilience_ai.simulator import BaselineReactiveStrategy, SimulationConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ResilienceAI strategy evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/evaluation"))
    args = parser.parse_args()

    report = run_benchmark(
        {"BaselineReactive": BaselineReactiveStrategy},
        SimulationConfig(),
        scenario_modes=(
            ScenarioMode.KNOWN_SHUTDOWN,
            ScenarioMode.SURPRISE_SHUTDOWN,
        ),
        seeds=(args.seed,),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary_metrics.csv"
    trajectory_path = args.output / "daily_trajectories.csv"
    report.summary.to_csv(summary_path, index=False)
    report.daily_trajectories.to_csv(trajectory_path, index=False)

    print(f"Wrote {summary_path}")
    print(f"Wrote {trajectory_path}")
    print(report.summary[
        ["strategy", "scenario_mode", "seed", "same_day_fill_rate",
         "ending_backlog", "procurement_expenditure_usd", "runtime_ms"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
