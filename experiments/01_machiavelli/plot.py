"""
Plotting for Machiavelli benchmark results using src.viz.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from src.viz import (
    ANTHRO_BLUE_500, ANTHRO_RED_500, ANTHRO_GREEN_500, ANTHRO_VIOLET_500,
    ANTHRO_AQUA_500, ANTHRO_YELLOW_500, ANTHRO_ORANGE_500, ANTHRO_FIG,
    ANTHRO_GRAY_400,
    plot_labeled_scatter, plot_hierarchical_bars,
)

MODEL_COLORS = [
    ANTHRO_BLUE_500, ANTHRO_RED_500, ANTHRO_GREEN_500,
    ANTHRO_VIOLET_500, ANTHRO_AQUA_500, ANTHRO_YELLOW_500,
    ANTHRO_ORANGE_500, ANTHRO_FIG,
]


POLICIES = ["standard", "good", "maxscore"]
POLICY_PREFIXES = {f"{p}_": p for p in POLICIES if p != "standard"}


def _parse_label(label: str) -> tuple[str, str]:
    """Extract (policy, model) from a directory label like 'good_claude-haiku-4-5'."""
    for prefix, policy in POLICY_PREFIXES.items():
        if label.startswith(prefix):
            return policy, label[len(prefix):]
    return "standard", label


def _load_all_results(output_dir: Path) -> pd.DataFrame | None:
    """Load all results.csv from model subdirectories."""
    dfs = []
    for f in sorted(output_dir.glob("*/results.csv")):
        label = f.parent.name
        df = pd.read_csv(f)
        policy, model = _parse_label(label)
        df["policy"] = policy
        df["model"] = model
        df["label"] = label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None


def generate_all_plots(output_dir: Path):
    df = _load_all_results(output_dir)
    if df is None or len(df) == 0:
        print("No results found for plotting.")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    skip = {"game", "agent", "traj_dir", "model", "policy", "label", "steps", "achievements", "points"}
    metric_cols = [c for c in df.columns if c not in skip]
    agg = df.groupby(["label", "model", "policy"])[["points"] + metric_cols].mean().reset_index()

    _plot_pareto(agg, plots_dir)
    _plot_key_metrics(agg, plots_dir)
    _plot_power_breakdown(agg, plots_dir)
    _plot_violations_breakdown(agg, plots_dir)
    print(f"\nPlots saved to {plots_dir}/")


POLICY_DISPLAY = {"standard": "Standard", "good": "Good", "maxscore": "Max Score"}


def _policy_row_labels(agg: pd.DataFrame) -> list[str]:
    """Return row labels for policies present in the data."""
    present = [p for p in POLICIES if p in agg["policy"].values]
    return [f"Policy: {POLICY_DISPLAY[p]}" for p in present]


def _split_by_policy(agg: pd.DataFrame, metrics: dict) -> dict:
    """Build bar data with one split per policy, using model name as group."""
    present = [p for p in POLICIES if p in agg["policy"].values]
    bar_data = {POLICY_DISPLAY[p]: {} for p in present}
    for _, row in agg.iterrows():
        cat_data = {}
        for metric_key, metric_name in metrics.items():
            val = row.get(metric_key, 0) or 0
            cat_data[metric_name] = (val, val, val)
        split = POLICY_DISPLAY[row["policy"]]
        bar_data[split][row["model"]] = cat_data
    return bar_data


def _plot_pareto(agg: pd.DataFrame, plots_dir: Path):
    """Score vs total harm scatter."""
    points = []
    for i, (_, row) in enumerate(agg.iterrows()):
        harm = (row.get("power.Σ", 0) or 0) + (row.get("violations.Σ", 0) or 0) + (row.get("utility.Σ", 0) or 0)
        score = row.get("game.score", 0) or 0
        points.append({
            "x": harm,
            "y": score,
            "label": row["label"],
            "color": MODEL_COLORS[i % len(MODEL_COLORS)],
            "marker": {"standard": "o", "good": "s", "maxscore": "D"}.get(row["policy"], "o"),
        })

    plot_labeled_scatter(
        points=points,
        xlabel="Total Harm (% of random)",
        ylabel="Game Score (% of random)",
        title="MACHIAVELLI: Score vs Harm",
        figsize=(7, 5),
        hlines=[{"y": 100, "color": ANTHRO_GRAY_400, "linestyle": "--", "alpha": 0.5, "label": "Random baseline"}],
        vlines=[{"x": 300, "color": ANTHRO_GRAY_400, "linestyle": "--", "alpha": 0.5}],
        save_path=str(plots_dir / "pareto.png"),
    )
    print("  Saved pareto.png")


def _plot_key_metrics(agg: pd.DataFrame, plots_dir: Path):
    """Bar chart of key metrics, split into Standard vs Good rows."""
    metrics = {"game.score": "Score", "power.Σ": "Power", "violations.Σ": "Violations", "utility.Σ": "Suffering"}
    available = {k: v for k, v in metrics.items() if k in agg.columns}
    if not available:
        return

    bar_data = _split_by_policy(agg, available)

    plot_hierarchical_bars(
        data=bar_data,
        title="MACHIAVELLI: Key Metrics",
        ylabel="% of random baseline",
        ylim=None,
        figsize=(7, 3.5),
        bar_width=0.18,
        category_order=["Score", "Power", "Violations", "Suffering"],
        splits_per_row=1,
        row_labels=_policy_row_labels(agg),
        row_ylabel="% of random baseline",
        show_error_bars=False,
        legend_loc="below",
        keep_first_legend=True,
        save_path=str(plots_dir / "key_metrics.png"),
        show_values=True,
        hlines=[{"y": 100, "color": ANTHRO_GRAY_400, "linestyle": "--", "alpha": 0.5}],
    )
    print("  Saved key_metrics.png")


def _plot_power_breakdown(agg: pd.DataFrame, plots_dir: Path):
    """Power subcategories grouped bars."""
    cols = [c for c in agg.columns if c.startswith("power.") and c != "power.Σ"]
    if not cols:
        return

    metrics = {c: c.replace("power.", "") for c in cols}
    bar_data = _split_by_policy(agg, metrics)

    plot_hierarchical_bars(
        data=bar_data,
        title="Power-Seeking Subcategories",
        ylabel="% of random baseline",
        ylim=None,
        figsize=(7, 3.5),
        bar_width=0.18,
        splits_per_row=1,
        row_labels=_policy_row_labels(agg),
        row_ylabel="% of random baseline",
        show_error_bars=False,
        legend_loc="below",
        keep_first_legend=True,
        save_path=str(plots_dir / "power_breakdown.png"),
        show_values=True,
        hlines=[{"y": 100, "color": ANTHRO_GRAY_400, "linestyle": "--", "alpha": 0.5}],
    )
    print("  Saved power_breakdown.png")


def _plot_violations_breakdown(agg: pd.DataFrame, plots_dir: Path):
    """Top ethical violation types."""
    v_cols = [c for c in agg.columns if c.startswith("violations.") and c != "violations.Σ"]
    if not v_cols:
        return

    mean_vals = agg[v_cols].mean()
    top = mean_vals[mean_vals > 0].nlargest(8).index.tolist()
    if not top:
        return

    metrics = {c: c.replace("violations.", "") for c in top}
    bar_data = _split_by_policy(agg, metrics)

    plot_hierarchical_bars(
        data=bar_data,
        title="Top Ethical Violations",
        ylabel="% of random baseline",
        ylim=None,
        bar_width=0.18,
        splits_per_row=1,
        row_labels=_policy_row_labels(agg),
        row_ylabel="% of random baseline",
        show_error_bars=False,
        legend_loc="below",
        keep_first_legend=True,
        figsize=(10, 3.5),
        rotate_xticks=45,
        save_path=str(plots_dir / "violations_breakdown.png"),
        show_values=True,
        hlines=[{"y": 100, "color": ANTHRO_GRAY_400, "linestyle": "--", "alpha": 0.5}],
    )
    print("  Saved violations_breakdown.png")
