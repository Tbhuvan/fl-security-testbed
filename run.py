#!/usr/bin/env python3
"""
FL Security Testbed — main entry point.

Usage:
  python run.py                           # Run baseline (no attack)
  python run.py --experiment byzantine    # Byzantine sweep
  python run.py --experiment defenses     # Defense comparison
  python run.py --experiment attacks      # Attack type sweep
  python run.py --experiment all          # All sweeps
  python run.py --list                    # List available experiments
"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("experiments/fl_testbed.log"),
    ],
)
log = logging.getLogger(__name__)

from experiments.config import (
    BASELINE,
    BYZANTINE_SWEEP,
    DEFENSE_COMPARISON,
    ATTACK_TYPE_SWEEP,
)
from experiments.runner import run_experiment, run_sweep

EXPERIMENTS = {
    "baseline":  ([BASELINE], "Single run: FedAvg, no attack"),
    "byzantine": (BYZANTINE_SWEEP, "FedAvg under increasing Byzantine fraction (0–40%)"),
    "defenses":  (DEFENSE_COMPARISON, "Compare FedAvg vs Krum vs TrimmedMean vs Median at 30% Byzantine"),
    "attacks":   (ATTACK_TYPE_SWEEP, "Compare attack types at 20% Byzantine with FedAvg"),
    "all":       (BYZANTINE_SWEEP + DEFENSE_COMPARISON + ATTACK_TYPE_SWEEP, "All sweeps"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="FL Security Testbed")
    parser.add_argument(
        "--experiment", "-e",
        default="baseline",
        choices=list(EXPERIMENTS.keys()),
        help="Which experiment set to run",
    )
    parser.add_argument("--list", action="store_true", help="List experiments and exit")
    parser.add_argument(
        "--results-dir",
        default="experiments/results",
        help="Directory for result JSON files",
    )
    args = parser.parse_args()

    if args.list:
        print("\nAvailable experiments:")
        for name, (configs, desc) in EXPERIMENTS.items():
            print(f"  {name:<12} — {desc} ({len(configs)} run(s))")
        return

    configs, desc = EXPERIMENTS[args.experiment]
    log.info("Running: %s — %s (%d experiment(s))", args.experiment, desc, len(configs))

    results_dir = Path(args.results_dir)

    if len(configs) == 1:
        run_experiment(configs[0], results_dir)
    else:
        run_sweep(configs, results_dir)


if __name__ == "__main__":
    main()
