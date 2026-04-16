"""
Machiavelli benchmark evaluation.

Uses src.tooling.complete() — any supported model works out of the box.

Usage:
    python experiments/01_machiavelli/run.py --model claude-haiku-4-5 --games avatar-of-the-wolf
    python experiments/01_machiavelli/run.py --model claude-haiku-4-5 claude-sonnet-4-6 claude-opus-4-6
    python experiments/01_machiavelli/run.py --model claude-sonnet-4-6 --policy good
    python experiments/01_machiavelli/run.py --model claude-sonnet-4-6 --policy maxscore
    python experiments/01_machiavelli/run.py --plot-only
"""

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "machiavelli"))
sys.path.insert(0, str(EXPERIMENT_DIR))

import agent as agent_mod
import evaluate as eval_mod
import plot as plot_mod
from machiavelli.config import MachiavelliConfig

cfg = MachiavelliConfig()
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "01_machiavelli"


def _short_name(model: str) -> str:
    """Extract a short directory-safe name from a model ID."""
    # openrouter/meta-llama/llama-3.3-70b-instruct -> llama-3.3-70b-instruct
    # together/qwen/qwen3-32b -> qwen3-32b
    return model.split("/")[-1]


async def run_model(model, games, episodes, max_steps, concurrency, temperature, policy, verbose, output_dir):
    prefix = f"{policy}_" if policy != "standard" else ""
    label = f"{prefix}{_short_name(model)}"
    model_dir = output_dir / label
    sem = asyncio.Semaphore(concurrency)

    async def run_one(game, ep):
        async with sem:
            return await agent_mod.play_game(
                model=model, game=game, episode=ep,
                max_steps=max_steps, temperature=temperature,
                policy=policy, output_dir=model_dir, verbose=verbose,
            )

    tasks = [run_one(g, ep) for g in games for ep in range(episodes)]
    print(f"\n{'='*60}")
    print(f"  {label}: {len(tasks)} episodes ({len(games)} games x {episodes} ep, concurrency={concurrency})")
    print(f"{'='*60}")

    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        print(f"  {len(errors)} failures:")
        for e in errors[:5]:
            print(f"    {e}")
    return model_dir


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", nargs="+", default=["claude-haiku-4-5"])
    p.add_argument("--policy", type=str, default="standard",
                    choices=list(agent_mod.POLICY_TEMPLATES.keys()),
                    help="Policy prompt to use (standard, good, maxscore)")
    # Keep --good for backwards compat
    p.add_argument("--good", action="store_true", help="Shorthand for --policy good")
    p.add_argument("--games", nargs="+", default=None)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--concurrency", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--skip-plot", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true", help="Print model actions each step")
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    policy = "good" if args.good else args.policy

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    games = args.games or cfg.games_test

    if not args.plot_only:
        for model in args.model:
            model_dir = await run_model(
                model, games, args.episodes, args.max_steps,
                args.concurrency, args.temperature, policy, args.verbose, output_dir,
            )
            df = eval_mod.evaluate_trajectories(model_dir)
            if df is not None and len(df) > 0:
                prefix = f"{policy}_" if policy != "standard" else ""
                label = f"{prefix}{_short_name(model)}"
                df.to_csv(model_dir / "results.csv", index=False)
                eval_mod.print_summary(df, label=label)

    if not args.skip_plot:
        plot_mod.generate_all_plots(output_dir)


if __name__ == "__main__":
    asyncio.run(main())
