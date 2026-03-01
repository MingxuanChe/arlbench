"""
plot_alpha_results.py
=====================
Reads the saved evaluation outputs from ``plot_alpha_sweep.py`` and
produces publication-ready plots of five HPO-reliability metrics as a
function of the risk-aversion parameter α:

    1. Mean performance          E[S_i]
    2. Standard deviation        Std[S_i]
    3. Mean-variance score       E[S_i] − α · Var[S_i]
    4. Worst-case score          min(S_i)  (worst observed incumbent score)

Data sources — sweep pipeline (written by plot_alpha_sweep.py / dr_compare_methods.py):
    <sweep_dir>/alpha_<a>/comparison_summary.csv   – aggregate stats
    <sweep_dir>/alpha_<a>/incumbent_scores.csv     – per-run S_i values

Data sources — cluster pipeline (written by dr_cluster_aggregate.py):
    <cluster_dir>/comparison_summary_all.csv       – aggregate stats (all alphas)
    <cluster_dir>/incumbent_scores_a<a>.csv        – per-run S_i values per alpha

Usage
-----
# Sweep pipeline (legacy):
python examples/plot_alpha_results.py \\
    --sweep-dir results/alpha_sweep/ppo_acrobot \\
    --failure-threshold -150

# Cluster pipeline:
python examples/plot_alpha_results.py \\
    --cluster-dir results/cluster_eval/ppo_cc_acrobot_dr \\
    --env-config cc_acrobot_dr --algorithm ppo
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ── allow importing from dr_compare_methods (same directory) ─────────────────
_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

# ─── style ────────────────────────────────────────────────────────────────────

METHOD_COLORS = {
    "gpucb":  "#2196F3",
    "rahbo":  "#FF9800",
    "erahbo": "#4CAF50",
}
METHOD_LABELS = {
    "gpucb":  "GPUCB",
    "rahbo":  "RAHBO",
    "erahbo": "ERAHBO",
}

DEFAULT_SWEEP_DIR       = "results/alpha_sweep/ppo_acrobot"
DEFAULT_FAILURE_THRESH  = -150.0
DEFAULT_RESULTS_ROOT    = "/home/mingxuan/Repos/Efficient-Risk-Averse-BO/results_dr"


# ─── source-score back-fill (for old CSVs without transfer_gap) ───────────────

def _resolve_trace_dir(
    root: Path,
    alpha: float,
    method: str,
    algorithm: str,
    env_config: str,
) -> Path | None:
    """Mirror of plot_alpha_sweep._trace_dir – resolve bo_trace directory."""
    alpha_dir = root / f"dr-dataset_a{alpha:.1f}"
    if not alpha_dir.is_dir():
        return None
    base = alpha_dir / method / "arlbench_dataset" / algorithm / env_config
    if method == "gpucb":
        return base
    for sub in (f"{alpha:.1f}", f"{alpha:g}", str(int(alpha))):
        candidate = base / sub
        if candidate.is_dir():
            return candidate
    return base / f"{alpha:.1f}"


def backfill_source_scores(
    scores_df: pd.DataFrame,
    results_root: Path,
    algorithm: str,
    env_config: str,
) -> pd.DataFrame:
    """Re-read bo_trace files to fill in missing source_score / transfer_gap.

    Used when *scores_df* was loaded from old-format CSVs that predate the
    source_score columns added in dr_compare_methods.py.  No RL training is
    re-run; only the bo_trace CSV files (already on disk) are read.
    """
    from dr_compare_methods import _extract_best_from_single_df  # noqa: PLC0415

    if scores_df.empty:
        return scores_df

    df = scores_df.copy()
    if "source_score" not in df.columns:
        df["source_score"] = float("nan")
    if "transfer_gap" not in df.columns:
        df["transfer_gap"] = float("nan")

    for (alpha, method), grp in df.groupby(["alpha", "method"]):
        trace_dir = _resolve_trace_dir(results_root, alpha, method, algorithm, env_config)
        if trace_dir is None or not trace_dir.is_dir():
            print(f"  [WARN] bo_trace dir not found: {trace_dir}", file=sys.stderr)
            continue

        trace_files = sorted(trace_dir.glob("bo_trace_run*.csv"))
        if not trace_files:
            trace_files = sorted(trace_dir.glob("*/bo_trace_run*.csv"))
        if not trace_files:
            print(f"  [WARN] no bo_trace_run*.csv in {trace_dir}", file=sys.stderr)
            continue

        print(f"  Backfilling source scores: α={alpha:g} / {method} ({len(trace_files)} runs)")
        for i, f in enumerate(trace_files):
            bo_df = pd.read_csv(f)
            _, src = _extract_best_from_single_df(
                bo_df, method, alpha=float(alpha), beta=float(alpha), invert_log=True
            )
            mask = (
                (df["alpha"]         == alpha) &
                (df["method"]        == method) &
                (df["incumbent_idx"] == i)
            )
            if not mask.any():
                continue
            df.loc[mask, "source_score"] = src
            target_val = df.loc[mask, "score_S_i"].iloc[0]
            tgt = float(target_val)
            df.loc[mask, "transfer_gap"] = (
                (src - tgt) / abs(src) if src != 0.0 and not np.isnan(src) else float("nan")
            )

    return df


# ─── data loading ─────────────────────────────────────────────────────────────

def _parse_alpha_from_dir(d: Path) -> float | None:
    """Extract numeric α from a directory named 'alpha_<value>'."""
    try:
        return float(d.name.removeprefix("alpha_"))
    except ValueError:
        return None


def load_sweep_data(
    sweep_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan *sweep_dir* for completed alpha_* subdirectories and load results.

    Returns
    -------
    summary_df : pd.DataFrame
        One row per (alpha, method).
        Columns: alpha, method, hpo_mean, hpo_std, hpo_var, hpo_min, hpo_max,
                 hpo_mean_var_score, hpo_worst_case_score,
                 n_incumbents.
    scores_df  : pd.DataFrame
        One row per (alpha, method, incumbent_idx).
        Columns: alpha, method, incumbent_idx, score_S_i.
    """
    summary_rows: list[dict] = []
    score_rows:   list[dict] = []

    alpha_dirs = sorted(
        [d for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith("alpha_")],
        key=lambda d: _parse_alpha_from_dir(d) or 0.0,
    )

    if not alpha_dirs:
        raise FileNotFoundError(
            f"No 'alpha_*' subdirectories found in '{sweep_dir}'.\n"
            "Run plot_alpha_sweep.py first to generate results."
        )

    for alpha_dir in alpha_dirs:
        alpha = _parse_alpha_from_dir(alpha_dir)
        if alpha is None:
            continue

        # ── comparison_summary.csv ────────────────────────────────────────────
        summary_path = alpha_dir / "comparison_summary.csv"
        if not summary_path.exists():
            print(f"  [WARN] missing {summary_path}, skipping α={alpha:g}")
            continue
        summary = pd.read_csv(summary_path)
        # Retroactively redefine worst-case as min(S_i) for old saved CSVs.
        if "hpo_min" in summary.columns:
            summary["hpo_worst_case_score"] = summary["hpo_min"]

        # ── incumbent_scores.csv  (wide: cols = method, method_source, method_transfer_gap, …)
        inc_path = alpha_dir / "incumbent_scores.csv"
        wide = pd.read_csv(inc_path) if inc_path.exists() else None

        for _, row in summary.iterrows():
            method = str(row["method"]).lower()
            n_inc  = int(row["n_incumbents"])

            if wide is not None and method in wide.columns:
                scores_vec   = wide[method].dropna().to_numpy(dtype=float)
                source_col   = f"{method}_source"
                gap_col      = f"{method}_transfer_gap"
                source_vec   = wide[source_col].to_numpy(dtype=float) if source_col in wide.columns else np.full(len(scores_vec), float("nan"))
                gap_vec      = wide[gap_col].to_numpy(dtype=float)    if gap_col   in wide.columns else np.full(len(scores_vec), float("nan"))

                n = min(len(scores_vec), len(source_vec), len(gap_vec))
                for run_idx in range(n):
                    score_rows.append({
                        "alpha":         alpha,
                        "method":        method,
                        "incumbent_idx": run_idx,
                        "score_S_i":     float(scores_vec[run_idx]),
                        "source_score":  float(source_vec[run_idx]),
                        "transfer_gap":  float(gap_vec[run_idx]),
                    })

            # Support both old name (hpo_mean_var_score) and new name (target_mean_var_score)
            mv_score = float(
                row.get("target_mean_var_score",
                row.get("hpo_mean_var_score", float("nan")))
            )
            summary_rows.append({
                "alpha":                  alpha,
                "method":                 method,
                "n_incumbents":           n_inc,
                "hpo_mean":               float(row["hpo_mean"]),
                "hpo_std":                float(row["hpo_std"]),
                "hpo_var":                float(row["hpo_var"]),
                "hpo_min":                float(row["hpo_min"]),
                "hpo_max":                float(row["hpo_max"]),
                "target_mean_var_score":  mv_score,
                "hpo_worst_case_score":   float(row["hpo_worst_case_score"]),
            })

        print(f"  Loaded α={alpha:g}  ({[r['method'] for r in summary_rows if r['alpha'] == alpha]})")

    summary_df = pd.DataFrame(summary_rows)
    scores_df  = pd.DataFrame(score_rows)
    return summary_df, scores_df


def load_cluster_data(cluster_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load results produced by ``dr_cluster_aggregate.py``.

    Reads
    -----
    ``<cluster_dir>/comparison_summary_all.csv``
        One row per (alpha, method).  Columns written by the cluster pipeline
        are normalised to the same names used by :func:`load_sweep_data`.

    ``<cluster_dir>/incumbent_scores_a<alpha>.csv``
        One row per (alpha, method, run_idx).  Provides per-incumbent S_i
        for violin / ridge plots.

    Returns
    -------
    Same ``(summary_df, scores_df)`` pair as :func:`load_sweep_data`.
    """
    summary_path = cluster_dir / "comparison_summary_all.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"comparison_summary_all.csv not found in '{cluster_dir}'.\n"
            "Run dr_cluster_aggregate.py first."
        )

    raw = pd.read_csv(summary_path)
    # Map cluster column names → plot_alpha_results internal names
    col_map = {
        "target_score_mean":       "hpo_mean",
        "target_score_std":        "hpo_std",
        "target_score_min":        "hpo_min",
        "target_score_max":        "hpo_max",
        "hpo_worst_case":          "hpo_worst_case_score",
        "target_mean_var_score":   "target_mean_var_score",  # keep as-is
    }
    raw = raw.rename(columns=col_map)
    # Add hpo_var from std
    if "hpo_var" not in raw.columns and "hpo_std" in raw.columns:
        raw["hpo_var"] = raw["hpo_std"] ** 2
    # Ensure backwards-compat alias
    if "target_mean_var_score" in raw.columns:
        raw["hpo_mean_var_score"] = raw["target_mean_var_score"]

    summary_rows = []
    for _, row in raw.iterrows():
        summary_rows.append({
            "alpha":                  float(row["alpha"]),
            "method":                 str(row["method"]).lower(),
            "n_incumbents":           int(row.get("n_incumbents", 1)),
            "hpo_mean":               float(row["hpo_mean"]),
            "hpo_std":                float(row.get("hpo_std", float("nan"))),
            "hpo_var":                float(row.get("hpo_var", float("nan"))),
            "hpo_min":                float(row.get("hpo_min", float("nan"))),
            "hpo_max":                float(row.get("hpo_max", float("nan"))),
            "target_mean_var_score":  float(row.get("target_mean_var_score", float("nan"))),
            "hpo_worst_case_score":   float(row.get("hpo_worst_case_score", float("nan"))),
        })
    summary_df = pd.DataFrame(summary_rows)

    # Per-incumbent scores from incumbent_scores_a<alpha>.csv files
    score_rows: list[dict] = []
    alphas = sorted(summary_df["alpha"].unique())
    for alpha in alphas:
        inc_path = cluster_dir / f"incumbent_scores_a{alpha:g}.csv"
        if not inc_path.exists():
            continue
        inc = pd.read_csv(inc_path)
        for _, row in inc.iterrows():
            score_rows.append({
                "alpha":         float(row["alpha"]),
                "method":        str(row["method"]).lower(),
                "incumbent_idx": int(row.get("run_idx", 0)),
                "score_S_i":     float(row.get("target_score", float("nan"))),
                "source_score":  float(row.get("source_score", float("nan"))),
                "transfer_gap":  float(row.get("transfer_gap", float("nan"))),
            })
        print(f"  Loaded α={alpha:g}  ({sorted(inc['method'].unique().tolist())})")

    scores_df = pd.DataFrame(score_rows)
    return summary_df, scores_df


# ─── plotting ─────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    print(f"  saved → {output_dir / stem}.{{pdf,png}}")


def _ax_style(ax: plt.Axes) -> None:
    ax.set_facecolor("#f7f7f7")
    ax.grid(axis="y", color="white", linewidth=1.1, zorder=0)
    ax.grid(axis="x", color="white", linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def plot_transfer_gap_violin(
    scores_df: pd.DataFrame,
    env_config: str,
    algorithm: str,
    output_dir: Path,
) -> None:
    """Violin + strip plot of relative transfer degradation per method.

    Relative degradation  =  (source_score − target_S_i) / |source_score|

    A value of 0.10 means target performance is 10 % worse than the
    source-domain score obtained during BO.  Positive = worse on target;
    zero = perfect transfer; negative = surprising improvement on target.

    One column per method, rows correspond to different α values (or a single
    row if only one α is present).  Each violin shows the distribution of
    N per-run degradation values.
    """
    if scores_df.empty or "transfer_gap" not in scores_df.columns:
        print("  [SKIP] no transfer_gap data available.")
        return

    # Drop rows where source_score is NaN (YAML / standard-layout inputs)
    df = scores_df.dropna(subset=["transfer_gap"]).copy()
    if df.empty:
        print("  [SKIP] all transfer_gap values are NaN (source scores unavailable).")
        return

    methods  = [m for m in ["gpucb", "rahbo", "erahbo"] if m in df["method"].unique()]
    alphas   = sorted(df["alpha"].unique())
    n_alphas = len(alphas)
    n_methods = len(methods)

    fig, axes = plt.subplots(
        1, n_methods,
        figsize=(4.5 * n_methods, 5),
        sharey=True,
        squeeze=False,
    )
    fig.suptitle(
        f"Relative transfer degradation: (source − target) / |source|\n"
        f"{algorithm.upper()} / {env_config}  —  "
        f"each violin = N independent BO runs; > 0 means worse on target",
        fontsize=12, fontweight="bold",
    )

    rng = np.random.default_rng(0)

    for col_idx, method in enumerate(methods):
        ax    = axes[0, col_idx]
        color = METHOD_COLORS.get(method, "grey")
        _ax_style(ax)

        # reference line at zero (perfect transfer)
        ax.axhline(0, color="#888", linewidth=1.0, linestyle="--", zorder=1,
                   label="0 % degradation (perfect transfer)")

        data_per_alpha: list[np.ndarray] = []
        for alpha in alphas:
            vals = df[
                (df["alpha"]  == alpha) &
                (df["method"] == method)
            ]["transfer_gap"].dropna().to_numpy(dtype=float)
            data_per_alpha.append(vals)

        # violins
        valid_idx  = [i for i, v in enumerate(data_per_alpha) if len(v) > 1]
        valid_data = [data_per_alpha[i] for i in valid_idx]
        if valid_data:
            parts = ax.violinplot(
                valid_data,
                positions=valid_idx,
                showmedians=True, showextrema=True, widths=0.55,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color); pc.set_alpha(0.35)
            for key in ("cmins", "cmaxes", "cbars", "cmedians"):
                parts[key].set_color(color); parts[key].set_linewidth(1.5)

        # jittered dots
        for a_idx, vals in enumerate(data_per_alpha):
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.10, 0.10, len(vals))
            ax.scatter(
                a_idx + jitter, vals,
                color=color, s=22, alpha=0.60, edgecolors="none", zorder=4,
            )
            # mean marker
            ax.hlines(
                vals.mean(), a_idx - 0.22, a_idx + 0.22,
                color=color, linewidth=2.2, zorder=5,
            )

        ax.set_xticks(range(n_alphas))
        ax.set_xticklabels([f"α={a:g}" for a in alphas], fontsize=10)
        ax.set_title(
            METHOD_LABELS.get(method, method),
            fontsize=12, fontweight="bold", color=color,
        )
        ax.set_xlabel("α", fontsize=11)
        if col_idx == 0:
            ax.set_ylabel("(source − target) / |source|  (relative degradation)", fontsize=11)

        # annotate mean gap per alpha
        for a_idx, vals in enumerate(data_per_alpha):
            if len(vals) == 0:
                continue
            ax.text(
                a_idx, ax.get_ylim()[1],
                f"μ={vals.mean()*100:.1f}%",
                ha="center", va="bottom", fontsize=7.5, color=color,
            )

    axes[0, 0].legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    _save(fig, output_dir, "transfer_gap_violin")
    plt.close(fig)


def plot_score_ridge(
    scores_df: pd.DataFrame,
    env_config: str,
    algorithm: str,
    output_dir: Path,
) -> None:
    """Ridge plot showing the distribution of absolute target performance S_i
    across α values, one panel per method.

    Each α level is drawn as a filled KDE curve stacked vertically so that
    higher α values sit at the top.  This makes it easy to see how the entire
    score distribution shifts (mean, spread, tail) as risk-aversion increases.
    """
    from scipy.stats import gaussian_kde  # noqa: PLC0415

    if scores_df.empty or "score_S_i" not in scores_df.columns:
        print("  [SKIP] no score_S_i data for ridge plot.")
        return

    methods = [m for m in ["gpucb", "rahbo", "erahbo"] if m in scores_df["method"].unique()]
    alphas  = sorted(scores_df["alpha"].unique())          # low → high
    n_methods = len(methods)

    # x range covers all data
    all_vals = scores_df["score_S_i"].dropna().to_numpy(dtype=float)
    x_min, x_max = all_vals.min(), all_vals.max()
    pad  = (x_max - x_min) * 0.08
    x_grid = np.linspace(x_min - pad, x_max + pad, 400)

    # vertical spacing between ridges
    OVERLAP = 0.55        # fraction of ridge height that overlaps the one below
    BASE_HEIGHT = 1.0     # relative unit; actual density is rescaled to this

    fig, axes = plt.subplots(
        1, n_methods,
        figsize=(4.5 * n_methods, 1.5 + 1.4 * len(alphas)),
        sharey=True, sharex=True,
        squeeze=False,
    )
    fig.suptitle(
        f"Absolute target performance  S_i  across α values\n"
        f"{algorithm.upper()} / {env_config}",
        fontsize=12, fontweight="bold",
    )

    alpha_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(alphas)))

    for col_idx, method in enumerate(methods):
        ax    = axes[0, col_idx]
        color = METHOD_COLORS.get(method, "grey")
        _ax_style(ax)
        ax.set_title(METHOD_LABELS.get(method, method),
                     fontsize=12, fontweight="bold", color=color)
        ax.set_xlabel("Target score  S_i", fontsize=10)
        if col_idx == 0:
            ax.set_ylabel("α  (risk-aversion)", fontsize=10)

        ytick_positions, ytick_labels = [], []

        for a_idx, alpha in enumerate(alphas):
            base_y = a_idx * BASE_HEIGHT            # baseline for this ridge
            vals = scores_df[
                (scores_df["alpha"]  == alpha) &
                (scores_df["method"] == method)
            ]["score_S_i"].dropna().to_numpy(dtype=float)

            if len(vals) < 2:
                ytick_positions.append(base_y)
                ytick_labels.append(f"α={alpha:g}")
                continue

            # KDE
            try:
                kde  = gaussian_kde(vals, bw_method="silverman")
                dens = kde(x_grid)
            except Exception:
                ytick_positions.append(base_y)
                ytick_labels.append(f"α={alpha:g}")
                continue

            # normalise so the peak reaches BASE_HEIGHT * (1 + OVERLAP)
            ridge_h = BASE_HEIGHT * (1.0 + OVERLAP)
            dens_scaled = dens / dens.max() * ridge_h if dens.max() > 0 else dens

            c = alpha_colors[a_idx]
            ax.fill_between(
                x_grid, base_y, base_y + dens_scaled,
                color=c, alpha=0.55, linewidth=0,
            )
            ax.plot(
                x_grid, base_y + dens_scaled,
                color=c, linewidth=1.5,
            )
            # baseline
            ax.axhline(base_y, color="#aaa", linewidth=0.5, zorder=0)
            # mean tick
            mean_val = vals.mean()
            ax.axvline(
                mean_val,
                ymin=(base_y) / (len(alphas) * BASE_HEIGHT + BASE_HEIGHT * OVERLAP),
                ymax=(base_y + dens_scaled[np.argmin(np.abs(x_grid - mean_val))]) /
                     (len(alphas) * BASE_HEIGHT + BASE_HEIGHT * OVERLAP),
                color=c, linewidth=1.8, linestyle="--", alpha=0.85, zorder=5,
            )
            ax.text(
                mean_val, base_y - 0.07,
                f"{mean_val:.0f}",
                ha="center", va="top", fontsize=7, color=c,
            )

            ytick_positions.append(base_y)
            ytick_labels.append(f"α={alpha:g}")

        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels, fontsize=9)
        ax.tick_params(axis="y", length=0)
        # remove top/right spines for cleanliness
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    fig.tight_layout()
    _save(fig, output_dir, "score_ridge")
    plt.close(fig)


def plot_aggregated_metrics(
    summary_df: pd.DataFrame,
    scores_df:  pd.DataFrame,
    failure_threshold: float,
    env_config: str,
    algorithm: str,
    output_dir: Path,
) -> None:
    """5-panel figure: one panel per HPO-reliability metric vs α."""

    methods  = [m for m in ["gpucb", "rahbo", "erahbo"] if m in summary_df["method"].unique()]
    alphas   = sorted(summary_df["alpha"].unique())
    n_alphas = len(alphas)

    metrics = [
        # (column_in_summary_df, y-axis label, panel title)
        ("hpo_mean",
         "E[S_i]",
         "Mean performance"),
        ("hpo_std",
         "Std[S_i]",
         "Standard deviation"),
        ("target_mean_var_score",
         "E[S_i] − α·Var[S_i]",
         "Mean-variance score (target)"),
        ("hpo_worst_case_score",
         "Worst-case  min(S_i)",
         "Worst-case score  min(S_i)"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.8 * len(metrics), 5))
    fig.suptitle(
        f"HPO reliability vs risk-aversion α  ·  "
        f"{algorithm.upper()} / {env_config}",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, (col, ylabel, title) in zip(axes, metrics):
        _ax_style(ax)

        for method in methods:
            sub = summary_df[summary_df["method"] == method].sort_values("alpha")
            if sub.empty:
                continue
            color = METHOD_COLORS.get(method, "grey")
            label = METHOD_LABELS.get(method, method)

            xs = sub["alpha"].to_numpy()
            ys = sub[col].to_numpy(dtype=float)

            ax.plot(xs, ys, marker="o", color=color, linewidth=2.2,
                    label=label, zorder=5)

            # ── overlay individual S_i dots on mean-performance panel ─────────
            if col == "hpo_mean" and not scores_df.empty:
                for a in alphas:
                    sub_s = scores_df[
                        (scores_df["alpha"]  == a) &
                        (scores_df["method"] == method)
                    ]
                    if sub_s.empty:
                        continue
                    jitter = np.random.default_rng(
                        int(a * 10) + hash(method) % 100
                    ).uniform(-0.02, 0.02, len(sub_s))
                    ax.scatter(
                        a + jitter,
                        sub_s["score_S_i"].to_numpy(),
                        color=color, s=14, alpha=0.28, zorder=3,
                    )

            # ── error band (±1 SE) for mean and std panels ───────────────────
            if col in ("hpo_mean", "hpo_std") and not scores_df.empty:
                se_list = []
                for a in xs:
                    sub_s = scores_df[
                        (scores_df["alpha"]  == a) &
                        (scores_df["method"] == method)
                    ]
                    vals = sub_s["score_S_i"].to_numpy() if not sub_s.empty else np.array([])
                    n    = len(vals)
                    if n > 1:
                        if col == "hpo_mean":
                            se_list.append(vals.std() / np.sqrt(n))
                        else:
                            # std of each run's score ← not directly available;
                            # use bootstrapped SE of the std estimate
                            rng2 = np.random.default_rng(42)
                            boot = [rng2.choice(vals, size=n, replace=True).std()
                                    for _ in range(200)]
                            se_list.append(np.std(boot))
                    else:
                        se_list.append(0.0)
                se_arr = np.array(se_list)
                if np.any(se_arr > 0):
                    ax.fill_between(
                        xs, ys - se_arr, ys + se_arr,
                        color=color, alpha=0.15, zorder=2,
                    )

        # ── axis formatting ───────────────────────────────────────────────────
        if n_alphas > 1:
            ax.set_xticks(alphas)
        ax.set_xlabel("α", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, loc="best", framealpha=0.9)

    fig.tight_layout()
    _save(fig, output_dir, "aggregated_metrics_vs_alpha")
    plt.close(fig)


def plot_score_distributions(
    scores_df:  pd.DataFrame,
    failure_threshold: float,
    env_config: str,
    algorithm: str,
    output_dir: Path,
) -> None:
    """Violin + strip chart of S_i distributions per method, faceted by α."""
    if scores_df.empty:
        return

    methods  = [m for m in ["gpucb", "rahbo", "erahbo"] if m in scores_df["method"].unique()]
    alphas   = sorted(scores_df["alpha"].unique())
    n_alphas = len(alphas)
    n_methods = len(methods)

    fig, axes = plt.subplots(
        n_methods, 1,
        figsize=(3.2 * n_alphas, 4.0 * n_methods),
        sharey=True,
        squeeze=False,
    )
    fig.suptitle(
        f"S_i distribution per α  ·  {algorithm.upper()} / {env_config}\n"
        f"(violin = density, dots = per-run scores, dashed = failure threshold {failure_threshold:.0f})",
        fontsize=12, fontweight="bold",
    )

    rng = np.random.default_rng(0)
    for row_idx, method in enumerate(methods):
        ax    = axes[row_idx, 0]
        color = METHOD_COLORS.get(method, "grey")
        _ax_style(ax)
        ax.axhline(failure_threshold, color="red", linewidth=1.0,
                   linestyle="--", alpha=0.6, zorder=1, label=f"threshold ({failure_threshold:.0f})")

        data_per_alpha = []
        for alpha in alphas:
            vals = scores_df[
                (scores_df["alpha"]  == alpha) &
                (scores_df["method"] == method)
            ]["score_S_i"].dropna().to_numpy(dtype=float)
            data_per_alpha.append(vals)

        # violin
        if any(len(v) > 1 for v in data_per_alpha):
            valid_idx  = [i for i, v in enumerate(data_per_alpha) if len(v) > 1]
            valid_data = [data_per_alpha[i] for i in valid_idx]
            parts = ax.violinplot(
                valid_data,
                positions=[i for i in valid_idx],
                showmedians=True, showextrema=True, widths=0.6,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color); pc.set_alpha(0.35)
            for key in ("cmins", "cmaxes", "cbars", "cmedians"):
                parts[key].set_color(color); parts[key].set_linewidth(1.5)

        # strip
        for a_idx, vals in enumerate(data_per_alpha):
            jitter = rng.uniform(-0.10, 0.10, len(vals))
            fails   = vals < failure_threshold
            ax.scatter(a_idx + jitter[~fails], vals[~fails],
                       color=color, s=22, alpha=0.60, edgecolors="none", zorder=4)
            ax.scatter(a_idx + jitter[fails],  vals[fails],
                       color="red",   s=22, alpha=0.55, edgecolors="none",
                       marker="x", zorder=4)

        ax.set_xticks(range(n_alphas))
        ax.set_xticklabels([f"α={a:g}" for a in alphas], fontsize=10)
        ax.set_title(METHOD_LABELS.get(method, method),
                     fontsize=12, fontweight="bold", color=color)
        ax.set_ylabel("S_i", fontsize=11)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.85)

    axes[-1, 0].set_xlabel("α", fontsize=12)
    fig.tight_layout()
    _save(fig, output_dir, "score_distributions_vs_alpha")
    plt.close(fig)


# ─── summary table ────────────────────────────────────────────────────────────

def print_table(
    summary_df: pd.DataFrame,
    methods: list[str],
) -> None:
    col_w = 13
    for alpha in sorted(summary_df["alpha"].unique()):
        sub = summary_df[summary_df["alpha"] == alpha]
        print(f"\n  ── α = {alpha:g} ──────────────────────────────────────────")
        header = f"  {'Metric':<33}" + "".join(
            f"{METHOD_LABELS.get(m, m):>{col_w}}" for m in methods
            if m in sub["method"].values
        )
        present = [m for m in methods if m in sub["method"].values]
        print(header)
        print("  " + "─" * (len(header) - 2))
        for col, label in [
            ("hpo_mean",               "Mean performance E[S_i]"),
            ("hpo_std",                "Std deviation Std[S_i]"),
            ("hpo_var",                "Variance Var[S_i]"),
            ("target_mean_var_score",  f"Mean-var  E−α·Var on target (α={alpha:g})"),
            ("hpo_worst_case_score",   "Worst-case  min(S_i)"),
            ("hpo_min",                "Min S_i"),
            ("hpo_max",                "Max S_i"),
        ]:
            row = f"  {label:<33}"
            for m in present:
                r = sub[sub["method"] == m]
                val = float(r[col].iloc[0]) if len(r) else float("nan")
                row += f"{val:>{col_w}.3f}"
            print(row)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot aggregated HPO-reliability metrics from alpha-sweep results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sweep-dir",         default=None,
                   help="Directory containing alpha_* subdirectories (sweep pipeline).")
    p.add_argument("--cluster-dir",       default=None,
                   help="Directory written by dr_cluster_aggregate.py "
                        "(contains comparison_summary_all.csv).")
    p.add_argument("--failure-threshold", type=float, default=DEFAULT_FAILURE_THRESH,
                   help="S_i below this value counts as a failure.")
    p.add_argument("--output-dir",        default=None,
                   help="Where to save plots (default: same as --sweep-dir).")
    p.add_argument("--env-config",        default="cc_acrobot_dr")
    p.add_argument("--algorithm",         default="ppo")
    p.add_argument(
        "--results-root", default=DEFAULT_RESULTS_ROOT, metavar="DIR",
        help=(
            "Path to the original bo_trace results root (the directory "
            "containing dr-dataset_a*/  subdirs).  Source scores are "
            "re-extracted from bo_trace CSVs when transfer_gap is unavailable "
            "in the saved incumbent_scores.csv files."
        ),
    )
    return p


def main() -> None:
    import matplotlib.ticker   # needed for PercentFormatter used inside plot fn

    args = build_parser().parse_args()

    if args.cluster_dir and args.sweep_dir:
        sys.exit("ERROR: specify either --cluster-dir or --sweep-dir, not both.")
    if not args.cluster_dir and not args.sweep_dir:
        # default: sweep dir for backwards compat
        args.sweep_dir = DEFAULT_SWEEP_DIR

    if args.cluster_dir:
        data_dir   = Path(args.cluster_dir)
        data_label = f"Cluster dir      : {data_dir}"
        loader     = lambda: load_cluster_data(data_dir)
    else:
        data_dir   = Path(args.sweep_dir)
        data_label = f"Sweep dir        : {data_dir}"
        loader     = lambda: load_sweep_data(data_dir)

    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(data_label)
    print(f"Output dir       : {output_dir}")
    print(f"Failure threshold: {args.failure_threshold}")
    print()

    summary_df, scores_df = loader()

    if summary_df.empty:
        print("No data loaded \u2013 nothing to plot.")
        return

    # ── Back-fill source scores from bo_trace files if needed ─────────────────
    need_backfill = (
        scores_df.empty or
        "transfer_gap" not in scores_df.columns or
        scores_df["transfer_gap"].isna().all()
    )
    if need_backfill and args.results_root:
        print(f"Backfilling source scores from bo_trace files in {args.results_root} ...")
        scores_df = backfill_source_scores(
            scores_df,
            results_root=Path(args.results_root),
            algorithm=args.algorithm,
            env_config=args.env_config,
        )
    elif need_backfill:
        print(
            "  Note: transfer_gap not available in saved CSVs.  "
            "Pass --results-root <bo_trace_root> to back-fill source scores.",
            file=sys.stderr,
        )

    methods = [m for m in ["gpucb", "rahbo", "erahbo"]
               if m in summary_df["method"].unique()]

    print_table(summary_df, methods)

    print("\nGenerating plots ...")
    plot_aggregated_metrics(
        summary_df, scores_df,
        failure_threshold=args.failure_threshold,
        env_config=args.env_config,
        algorithm=args.algorithm,
        output_dir=output_dir,
    )
    plot_score_distributions(
        scores_df,
        failure_threshold=args.failure_threshold,
        env_config=args.env_config,
        algorithm=args.algorithm,
        output_dir=output_dir,
    )
    plot_transfer_gap_violin(
        scores_df,
        env_config=args.env_config,
        algorithm=args.algorithm,
        output_dir=output_dir,
    )
    plot_score_ridge(
        scores_df,
        env_config=args.env_config,
        algorithm=args.algorithm,
        output_dir=output_dir,
    )

    print(f"\nDone.  Plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
