"""
Run the random agent across Machiavelli games and collect per-episode scores.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from machiavelli.agent.random_agent import RandomAgent
from machiavelli.game.machiavelli_env import MachiavelliEnv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(_PROJECT_ROOT / "machiavelli")


def run_random_agent(
    games: list[str],
    episodes: int = 5,
    max_steps: int = 500,
    output_dir: Path = Path("outputs"),
) -> pd.DataFrame:
    """Run RandomAgent on each game for N episodes, return per-episode scores."""
    agent = RandomAgent()
    rows = []

    total = len(games) * episodes
    with tqdm(total=total, desc="Random agent rollouts") as pbar:
        for game in games:
            for ep in range(episodes):
                try:
                    row = _play_one(agent, game, ep, max_steps)
                    rows.append(row)
                except Exception as e:
                    print(f"  Error on {game} ep{ep}: {e}")
                pbar.update(1)

    df = pd.DataFrame(rows)

    # Save trajectories as CSV (lightweight — no pkl needed for random agent)
    output_dir.mkdir(parents=True, exist_ok=True)
    return df


def _play_one(agent: RandomAgent, game: str, episode: int, max_steps: int) -> dict:
    """Play one episode with the random agent and return Machiavelli scores."""
    env = MachiavelliEnv(game=game, agent_name="random")
    obs, info = env.reset(seed=episode)
    done = False
    rew = 0
    step = 0

    while not done and step < max_steps:
        action = agent.get_action(obs, rew, done, info)
        obs, rew, done, info = env.step(action)
        step += 1

    traj = env.trajectory
    scores = traj.get_mach_scores()
    counts = traj.get_mach_counts()

    env.close()

    return {
        "game": game,
        "episode": episode,
        "steps": step,
        "points": traj.get_points(normalize=True),
        **{f"score.{k}": v for k, v in scores.items()},
        **{f"count.{k}": v for k, v in counts.items()},
    }


def print_summary(df: pd.DataFrame):
    """Print summary statistics of the random agent runs."""
    print(f"\n{'─'*60}")
    print(f"Random Agent: {len(df)} episodes across {df['game'].nunique()} games")
    print(f"{'─'*60}")

    for col, name in [
        ("score.game.score", "Score (% of random)"),
        ("score.power.Σ", "Power-seeking"),
        ("score.violations.Σ", "Violations"),
        ("score.utility.Σ", "Suffering"),
    ]:
        if col in df.columns:
            print(f"  {name:.<30} {df[col].mean():>7.1f}% ± {df[col].std():>5.1f}")

    print(f"  {'Steps (avg)':.<30} {df['steps'].mean():>7.0f}")
    print()
