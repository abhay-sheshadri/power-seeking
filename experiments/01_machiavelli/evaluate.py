"""Evaluate saved Machiavelli trajectories and compute behavioral scores."""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from machiavelli.game.machiavelli_env import Trajectory


def evaluate_trajectories(traj_dir: Path) -> pd.DataFrame | None:
    """Load .pkl trajectories and compute Machiavelli scores."""
    pkl_files = sorted(Path(traj_dir).glob("**/*.pkl"))
    if not pkl_files:
        print(f"No trajectory files found in {traj_dir}")
        return None

    rows = []
    for path in tqdm(pkl_files, desc=f"Evaluating {traj_dir.name}"):
        try:
            traj = Trajectory.load(path)
            rows.append({
                "game": traj.game_name,
                "agent": traj.agent_name,
                "points": traj.get_points(normalize=True),
                "achievements": len(traj.get_achievements()),
                "steps": len(traj),
                **traj.get_mach_scores(),
            })
        except Exception as e:
            print(f"  Error loading {path}: {e}")

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame, label: str = ""):
    """Print concise evaluation summary."""
    if df is None or len(df) == 0:
        return

    print(f"\n{'─'*60}")
    print(f"Results: {label} ({len(df)} episodes, {df['game'].nunique()} games)")
    print(f"{'─'*60}")

    for col, name in [("game.score", "Score (% of random)"), ("power.Σ", "Power-seeking"),
                       ("violations.Σ", "Violations"), ("utility.Σ", "Suffering")]:
        if col in df.columns:
            print(f"  {name:.<30} {df[col].mean():>7.1f}%")

    print(f"  {'Points (normalized)':.<30} {df['points'].mean():>7.1f}%")
    print(f"  {'Steps (avg)':.<30} {df['steps'].mean():>7.0f}")
    print()
