"""
Plot Score vs Power and Score vs Violations correlations under the random policy.

Includes:
- Overall scatter plots with trend lines (all episodes pooled)
- Per-game Spearman correlation bar charts (one rho per game)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from src.viz import (
    ANTHRO_BLUE_500, ANTHRO_RED_500, ANTHRO_GREEN_500, ANTHRO_GRAY_400,
    ANTHRO_CLAY, ANTHRO_VIOLET_500, _style_ax, _save_fig,
    plot_scatter_with_trend,
)


def generate_all_plots(output_dir: Path):
    csv_path = output_dir / "results.csv"
    if not csv_path.exists():
        print("No results.csv found — run simulation first.")
        return

    df = pd.read_csv(csv_path)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Overall scatter plots ---

    _plot_correlation(
        df, x_col="score.game.score", y_col="score.power.Σ",
        xlabel="Game Score (% of random)",
        ylabel="Power-Seeking (% of random)",
        title="Random Policy: Score vs Power",
        save_path=str(plots_dir / "score_vs_power.png"),
    )

    _plot_correlation(
        df, x_col="score.game.score", y_col="score.violations.Σ",
        xlabel="Game Score (% of random)",
        ylabel="Violations (% of random)",
        title="Random Policy: Score vs Violations",
        save_path=str(plots_dir / "score_vs_violations.png"),
    )

    _plot_correlation(
        df, x_col="count.game.score", y_col="count.power.Σ",
        xlabel="Game Score (raw points %)",
        ylabel="Power-Seeking (raw count)",
        title="Random Policy: Score vs Power (raw counts)",
        save_path=str(plots_dir / "score_vs_power_raw.png"),
    )

    _plot_correlation(
        df, x_col="count.game.score", y_col="count.violations.Σ",
        xlabel="Game Score (raw points %)",
        ylabel="Violations (raw count)",
        title="Random Policy: Score vs Violations (raw counts)",
        save_path=str(plots_dir / "score_vs_violations_raw.png"),
    )

    _plot_correlation(
        df, x_col="score.power.Σ", y_col="score.violations.Σ",
        xlabel="Power-Seeking (% of random)",
        ylabel="Violations (% of random)",
        title="Random Policy: Power vs Violations",
        save_path=str(plots_dir / "power_vs_violations.png"),
    )

    # --- Per-game correlation bar charts ---

    _plot_per_game_correlations(
        df,
        x_col="count.game.score", y_col="count.power.Σ",
        title="Per-Game Spearman ρ: Score vs Power",
        save_path=str(plots_dir / "per_game_score_vs_power.png"),
        bar_color=ANTHRO_BLUE_500,
    )

    _plot_per_game_correlations(
        df,
        x_col="count.game.score", y_col="count.violations.Σ",
        title="Per-Game Spearman ρ: Score vs Violations",
        save_path=str(plots_dir / "per_game_score_vs_violations.png"),
        bar_color=ANTHRO_RED_500,
    )

    _plot_per_game_correlations(
        df,
        x_col="count.power.Σ", y_col="count.violations.Σ",
        title="Per-Game Spearman ρ: Power vs Violations",
        save_path=str(plots_dir / "per_game_power_vs_violations.png"),
        bar_color=ANTHRO_VIOLET_500,
    )

    print(f"\nPlots saved to {plots_dir}/")


def _plot_correlation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str,
):
    """Plot a single scatter+trend, dropping NaN/zero-only columns gracefully."""
    if x_col not in df.columns or y_col not in df.columns:
        print(f"  Skipping {title}: missing columns")
        return

    mask = df[x_col].notna() & df[y_col].notna()
    x = df.loc[mask, x_col].values.astype(float)
    y = df.loc[mask, y_col].values.astype(float)

    if len(x) < 3:
        print(f"  Skipping {title}: too few data points ({len(x)})")
        return

    plot_scatter_with_trend(
        x=np.array(x),
        y=np.array(y),
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        alpha=0.4,
        point_size=25,
        jitter=(0.5, 0.5),
        save_path=save_path,
    )
    print(f"  Saved {Path(save_path).name}")


def _plot_per_game_correlations(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    save_path: str,
    bar_color: str = ANTHRO_BLUE_500,
):
    """Horizontal bar chart showing Spearman rho per game, sorted by rho."""
    if x_col not in df.columns or y_col not in df.columns:
        print(f"  Skipping {title}: missing columns")
        return

    records = []
    for game, gdf in df.groupby("game"):
        x = gdf[x_col].dropna().values.astype(float)
        y = gdf[y_col].dropna().values.astype(float)
        n = min(len(x), len(y))
        if n < 3:
            continue
        x, y = x[:n], y[:n]
        # Skip if either variable is constant (rho undefined)
        if np.std(x) == 0 or np.std(y) == 0:
            records.append({"game": game, "rho": 0.0, "p": 1.0, "n": n})
            continue
        rho, p = sp_stats.spearmanr(x, y)
        records.append({"game": game, "rho": rho, "p": p, "n": n})

    if not records:
        print(f"  Skipping {title}: no games with enough data")
        return

    rdf = pd.DataFrame(records).sort_values("rho", ascending=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, max(6, len(rdf) * 0.35)), dpi=150)
    fig.patch.set_facecolor("white")

    y_pos = np.arange(len(rdf))
    colors = [bar_color if p < 0.05 else ANTHRO_GRAY_400 for p in rdf["p"]]
    bars = ax.barh(y_pos, rdf["rho"].values, color=colors, edgecolor="none", height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(rdf["game"].values, fontsize=8)
    ax.set_xlabel("Spearman ρ", fontsize=11)
    ax.axvline(0, color="black", linewidth=0.8)

    # Annotate with rho values
    for i, (rho, p, n) in enumerate(zip(rdf["rho"], rdf["p"], rdf["n"])):
        sig = "*" if p < 0.05 else ""
        ax.text(
            rho + 0.02 if rho >= 0 else rho - 0.02,
            i,
            f"{rho:.2f}{sig}  (n={n})",
            va="center",
            ha="left" if rho >= 0 else "right",
            fontsize=7,
        )

    # Overall correlation across all episodes
    all_x = df[x_col].dropna().values.astype(float)
    all_y = df[y_col].dropna().values.astype(float)
    n_all = min(len(all_x), len(all_y))
    rho_all, p_all = sp_stats.spearmanr(all_x[:n_all], all_y[:n_all])

    ax.set_title(
        f"{title}\nOverall ρ={rho_all:.3f} (p={p_all:.3f}, N={n_all})  |  gray = p≥0.05",
        fontsize=11, fontweight="bold",
    )

    _style_ax(ax)
    ax.set_xlim(-1.1, 1.1)
    plt.tight_layout()
    _save_fig(fig, save_path)
    print(f"  Saved {Path(save_path).name}")
