"""
Machiavelli game-playing agent built on src.tooling.complete().

Plays Choose-Your-Own-Adventure games via LLM inference.
Supports any model in the tooling layer.
"""

import collections
import json
import os
import re
from pathlib import Path

import numpy as np

from machiavelli.game.machiavelli_env import MachiavelliEnv
from src.tooling import complete
from src.utils import load_prompt_file

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(_PROJECT_ROOT / "machiavelli")

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


POLICY_TEMPLATES = {
    "standard": "standard.jinja2",
    "good": "good.jinja2",
    "maxscore": "maxscore.jinja2",
}


def _build_system_prompt(info: dict, policy: str = "standard") -> str:
    achievements = []
    for ach, ach_info in info["game_achievements"].items():
        if ach not in info["game_state"]["unlocked_achievements"]:
            achievements.append(f"- {ach_info['description']} ({ach_info['points']} points)")
    ach_text = "\n".join(achievements) or "(all achievements unlocked)"

    template_name = POLICY_TEMPLATES.get(policy, "standard.jinja2")
    template = PROMPTS_DIR / template_name
    return load_prompt_file(
        template,
        title=info["title"],
        short_summary=info["short_summary"],
        player_character=info["player_character"],
        achievements_text=ach_text,
    )


def _strip_choices(obs: str) -> str:
    """Remove the trailing numbered choices that the env appends to obs."""
    lines = obs.strip().split("\n")
    while lines and re.match(r"^\d+:\s", lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def _parse_action(text: str, num_actions: int) -> int | None:
    """Extract action integer from <action> tags, falling back to first integer."""
    m = re.search(r"<action>\s*(\d+)\s*</action>", text)
    if m:
        val = int(m.group(1))
        if 0 <= val < num_actions:
            return val
    for n in re.findall(r"\d+", text.strip()):
        val = int(n)
        if 0 <= val < num_actions:
            return val
    return None


async def _get_action(
    model: str, obs: str, info: dict, policy: str,
    temperature: float, node_history: collections.deque,
) -> tuple[int, str, str]:
    """Get an action from the LLM. Returns (action, raw_response, user_msg)."""
    num_actions = info["game_state"]["num_choices"]

    # Loop detection
    for node in info["game_state"]["current_nodes"]:
        node_history.append(node)
    if len(node_history) > 5:
        last_5 = "".join(list(node_history)[-5:])
        earlier = "".join(list(node_history)[:-5])
        if last_5 in earlier:
            return int(np.random.choice(num_actions)), "[LOOP_DETECTED]", ""

    sys_prompt = _build_system_prompt(info, policy)
    scene = _strip_choices(obs)
    user_msg = load_prompt_file(
        PROMPTS_DIR / "turn.jinja2",
        scene=scene,
        actions=info["game_state"]["choice_texts"],
    )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_msg},
    ]
    raw = await complete(messages, model=model, temperature=temperature)
    action = _parse_action(raw, num_actions)

    if action is None:
        action = int(np.random.choice(num_actions))
        raw = f"[PARSE_FAILED] {raw}"

    return action, raw, user_msg


async def play_game(
    model: str, game: str, episode: int = 0, max_steps: int = 500,
    temperature: float = 0.2, policy: str = "standard", output_dir: Path = Path("outputs"),
    verbose: bool = False,
) -> dict:
    """Play one episode of a game. Saves trajectory, rollout, and transcript."""
    prefix = f"{policy}_" if policy != "standard" else ""
    short = model.split("/")[-1]
    label = f"{prefix}{short}"
    env = MachiavelliEnv(game=game, agent_name=label)
    obs, info = env.reset(seed=episode)
    node_history = collections.deque(maxlen=1000)

    rollout = []
    transcript = []  # Model's-eye view of the conversation
    rew, done, step, random_actions = 0, False, 0, 0

    sys_prompt = _build_system_prompt(info, policy)
    transcript.append({"role": "system", "content": sys_prompt})

    while not done and step < max_steps:
        action, raw, user_msg = await _get_action(model, obs, info, policy, temperature, node_history)
        if "[LOOP_DETECTED" in raw or "[PARSE_FAILED" in raw:
            random_actions += 1

        choices = info["game_state"]["choice_texts"]
        if verbose:
            print(f"\n--- [{game}] Step {step} ---", flush=True)
            print(f"Scene: {obs.strip()[:300]}...")
            for i, c in enumerate(choices):
                marker = " >>>" if i == action else "    "
                print(f"{marker} [{i}] {c[:80]}")
            print(f"Model: {raw[:200]}", flush=True)

        rollout.append({
            "step": step,
            "observation": obs.strip(),
            "choices": choices,
            "action": action,
            "action_text": choices[action] if action < len(choices) else "?",
            "model_response": raw,
            "reward": rew,
        })

        # Build transcript from model's perspective
        transcript.append({"role": "user", "content": user_msg})
        transcript.append({"role": "assistant", "content": raw})

        obs, rew, done, info = env.step(action)
        step += 1

    # Save outputs
    game_dir = output_dir / game
    game_dir.mkdir(parents=True, exist_ok=True)

    env.trajectory.save(game_dir / f"ep{episode}.pkl")

    with open(game_dir / f"ep{episode}_rollout.json", "w") as f:
        json.dump({
            "model": model, "game": game, "episode": episode, "policy": policy,
            "steps": step, "random_actions": random_actions,
            "final_points": env.trajectory.get_points(normalize=True),
            "achievements": env.trajectory.get_achievements(),
            "rollout": rollout,
        }, f, indent=2, default=str)

    with open(game_dir / f"ep{episode}_transcript.json", "w") as f:
        json.dump(transcript, f, indent=2, default=str)

    env.close()
    pts = env.trajectory.get_points(normalize=True)
    print(f"  [{game}] ep{episode}: {step} steps, {pts:.1f}% pts, {random_actions} random", flush=True)
    return {"game": game, "episode": episode, "steps": step, "points": pts}
