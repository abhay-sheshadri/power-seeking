"""
Correlation analysis: Score vs Power / Violations under the random policy.

Runs a random agent across all test games for multiple episodes, then
computes per-episode Machiavelli scores and plots Score vs Power and
Score vs Violations scatter plots with OLS trend lines and Spearman rho.

Usage:
    python experiments/02_random_correlations/run.py
    python experiments/02_random_correlations/run.py --games avatar-of-the-wolf battlemage
    python experiments/02_random_correlations/run.py --episodes 10
    python experiments/02_random_correlations/run.py --plot-only
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "machiavelli"))
sys.path.insert(0, str(EXPERIMENT_DIR))

import simulate
import plot as plot_mod
from machiavelli.config import MachiavelliConfig

cfg = MachiavelliConfig()
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "02_random_correlations"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", nargs="+", default=None)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    games = args.games or cfg.games_test

    if not args.plot_only:
        df = simulate.run_random_agent(games, args.episodes, args.max_steps, output_dir)
        if df is not None and len(df) > 0:
            df.to_csv(output_dir / "results.csv", index=False)
            simulate.print_summary(df)

    plot_mod.generate_all_plots(output_dir)


if __name__ == "__main__":
    main()
