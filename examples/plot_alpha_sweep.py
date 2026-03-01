"""
plot_alpha_sweep.py
===================
Full HPO-reliability comparison that sweeps the risk-aversion parameter
α ∈ {0.5, 1.0, 1.5, 2.0}.

For each α the script
  1. Resolves an incumbent per independent BO run for each method:
       GPUCB  → argmax mean_performance
       RAHBO  → argmax mean_performance − α · Var(seed returns)
       ERAHBO → argmax mean_performance − α · Std(seed returns)
  2. Trains the incumbent *from scratch* on a fixed set of held-out
     target domains (shared across all α to make comparison apples-to-apples).
  3. Records per-incumbent transfer scores S_i and summarises HPO
     reliability (hpo_mean, hpo_std, hpo_var, hpo_worst_case, …).

Then it plots how these metrics change as α increases.

Directory layout expected (Efficient-Risk-Averse-BO results)
------------------------------------------------------------
  <results_root>/
    dr-dataset_a{alpha}/
      gpucb/arlbench_dataset/{algorithm}/{env_config}/
        bo_trace_run*.csv
      rahbo/arlbench_dataset/{algorithm}/{env_config}/{alpha}/
        bo_trace_run*.csv
      erahbo/arlbench_dataset/{algorithm}/{env_config}/{alpha}/
        bo_trace_run*.csv

Usage
-----
# Default: ppo / cc_acrobot_dr, all four alpha values
python examples/plot_alpha_sweep.py

# Custom
python examples/plot_alpha_sweep.py \\
    --results-root /path/to/results_dr \\
    --algorithm ppo --env-config cc_acrobot_dr \\
    --alphas 0.5 1.0 1.5 2.0 \\
    --n-targets 20 --n-seeds 5 \\
    --output-dir results/alpha_sweep/ppo_acrobot

# Skip expensive RL runs and just re-plot from a previous saved JSON
python examples/plot_alpha_sweep.py \\
    --load-results results/alpha_sweep/ppo_acrobot/all_results.json
"""

from __future__ import annotations

import argparse
import json
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

# ── import the transfer-eval machinery from dr_compare_methods ────────────────
_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from dr_compare_methods import (   # noqa: E402
    run_comparison,
    save_comparison,
    _NumpyEncoder,
)
from dr_target_eval import (       # noqa: E402
    load_env_config,
    sample_target_domains,
)

# ─── constants ────────────────────────────────────────────────────────────────

DEFAULT_ROOT   = "/home/mingxuan/Repos/Efficient-Risk-Averse-BO/results_dr"
DEFAULT_ALGO   = "ppo"
DEFAULT_ENV    = "cc_acrobot_dr"
DEFAULT_ALPHAS = [0.5, 1.0, 1.5, 2.0]
DEFAULT_OUTPUT = "results/alpha_sweep/ppo_acrobot"

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

# ─── path resolution ──────────────────────────────────────────────────────────

def _trace_dir(
    root: Path,
    alpha: float,
    method: str,
    algorithm: str,
    env_config: str,
) -> Path:
    """Return the directory that contains bo_trace_run*.csv for this combo."""
    alpha_dir = root / f"dr-dataset_a{alpha:.1f}"
    base = alpha_dir / method / "arlbench_dataset" / algorithm / env_config

    if method == "gpucb":
        return base          # no alpha sub-folder for GPUCB

    # rahbo / erahbo store traces inside an alpha sub-folder (1.0, 1.5, …)
    # Try float form first ("1.0"), then integer form ("1")
    for sub in (f"{alpha:.1f}", f"{alpha:g}", str(int(alpha))):
        candidate = base / sub
        if candidate.is_dir():
            return candidate

    return base / f"{alpha:.1f}"   # will raise clearly if missing


def _method_path_for_alpha(
    root: Path,
    alpha: float,
    method: str,
    algorithm: str,
    env_config: str,
) -> str:
    return str(_trace_dir(root, alpha, method, algorithm, env_config))


# ─── per-alpha evaluation ─────────────────────────────────────────────────────

def evaluate_one_alpha(
    root: Path,
    alpha: float,
    methods: list[str],
    env_cfg: dict,
    algorithm: str,
    env_config: str,
    target_domains: list[dict],
    seed_list: list[int],
    n_total_timesteps: float | None,
    n_eval_steps: int | None,
    n_eval_episodes: int | None,
    output_dir: Path | None,
) -> dict[str, dict]:
    """Run the full dr_compare_methods evaluation for one α value.

    Returns the ``results`` dict produced by ``run_comparison``.
    GPUCB selection criterion is always max(mean_performance) regardless of α.
    RAHBO uses  mean − α · Var.
    ERAHBO uses mean − α · Std.
    """
    method_pairs: list[tuple[str, str]] = []
    for m in methods:
        path = _method_path_for_alpha(root, alpha, m, algorithm, env_config)
        p = Path(path)
        if not p.is_dir():
            print(f"  [WARN] directory not found for α={alpha} / {m}: {path}", file=sys.stderr)
            continue
        method_pairs.append((m, path))

    if not method_pairs:
        raise FileNotFoundError(f"No valid method directories found for α={alpha}.")

    alpha_out = output_dir / f"alpha_{alpha:g}" if output_dir else None

    # alpha is passed as both the RAHBO variance penalty and the ERAHBO std penalty
    # so they match the training-time criterion used to build the dataset
    results = run_comparison(
        methods=method_pairs,
        env_cfg=env_cfg,
        algorithm=algorithm,
        target_domains=target_domains,
        seed_list=seed_list,
        n_total_timesteps=n_total_timesteps,
        n_eval_steps=n_eval_steps,
        n_eval_episodes=n_eval_episodes,
        alpha=alpha,   # RAHBO selection:  mean − alpha * Var
        beta=alpha,    # ERAHBO selection: mean − alpha * Std
        output_dir=alpha_out,
    )

    if alpha_out:
        save_comparison(
            output_dir=alpha_out,
            results=results,
            target_domains=target_domains,
            seed_list=seed_list,
            env_cfg=env_cfg,
            algorithm=algorithm,
            alpha=alpha,
            beta=alpha,
        )

    return results


# ─── result serialisation ─────────────────────────────────────────────────────

def _results_to_serialisable(
    all_results: dict[float, dict[str, dict]],
) -> dict:
    out: dict = {}
    for alpha, method_results in all_results.items():
        out[str(alpha)] = {}
        for method, r in method_results.items():
            r2 = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
            r2["incumbent_scores"] = r["incumbent_scores"].tolist()
            r2["source_scores"]    = r["source_scores"].tolist()
            r2["all_rewards"]      = r["all_rewards"].tolist()
            r2["incumbents"]       = r.get("incumbents", [])
            out[str(alpha)][method] = r2
    return out


def save_all_results(
    all_results: dict[float, dict[str, dict]],
    path: Path,
) -> None:
    with open(path, "w") as f:
        json.dump(_results_to_serialisable(all_results), f, indent=2, cls=_NumpyEncoder)
    print(f"All results saved → {path}")


def load_all_results(path: Path) -> dict[float, dict[str, dict]]:
    with open(path) as f:
        raw = json.load(f)
    out: dict[float, dict[str, dict]] = {}
    for alpha_str, method_results in raw.items():
        alpha = float(alpha_str)
        out[alpha] = {}
        for method, r in method_results.items():
            r2 = dict(r)
            r2["incumbent_scores"] = np.array(r["incumbent_scores"])
            r2["source_scores"]    = np.array(r.get("source_scores", [float("nan")]))
            r2["all_rewards"]      = np.array(r["all_rewards"])
            out[alpha][method] = r2
    return out


# ─── plotting ─────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    print(f"  saved → {output_dir / stem}.{{pdf,png}}")


def _build_sweep_df(
    all_results: dict[float, dict[str, dict]],
    methods: list[str],
) -> pd.DataFrame:
    rows = []
    for alpha in sorted(all_results):
        for method in methods:
            r = all_results[alpha].get(method)
            if r is None:
                continue
            rows.append({
                "alpha":              alpha,
                "method":             method,
                "hpo_mean":           r["hpo_mean"],
                "hpo_std":            r["hpo_std"],
                "hpo_var":            r["hpo_var"],
                "hpo_min":            r["hpo_min"],
                "hpo_max":            r["hpo_max"],
                "hpo_mean_var_score": r["hpo_mean_var_score"],
                "hpo_worst_case":     r["hpo_worst_case_score"],
            })
    return pd.DataFrame(rows)


def plot_metrics_vs_alpha(
    all_results: dict[float, dict[str, dict]],
    methods: list[str],
    env_config: str,
    algorithm: str,
    output_dir: Path,
) -> None:
    """Line plots of each HPO-reliability metric vs α, one line per method."""
    df     = _build_sweep_df(all_results, methods)
    alphas = sorted(all_results.keys())

    metrics = [
        ("hpo_mean",             "E[S_i]  mean transfer perf."),
        ("hpo_std",              "Std[S_i]  HPO instability"),
        ("hpo_var",              "Var[S_i]  HPO variance"),
        ("hpo_worst_case",       "E[S_i] − α·Std  (worst-case score)"),
        ("hpo_mean_var_score",   "E[S_i] − α·Var  (mean-var score)"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 5))
    fig.suptitle(
        f"HPO reliability vs α  |  {algorithm.upper()} / {env_config}  "
        f"(target-domain transfer, N=20 runs per method)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    for ax, (metric, ylabel) in zip(axes, metrics):
        ax.set_facecolor("#f7f7f7")
        ax.grid(axis="y", color="white", linewidth=1.0, zorder=0)
        ax.grid(axis="x", color="white", linewidth=0.6, zorder=0)

        for method in methods:
            sub = df[df["method"] == method].sort_values("alpha")
            if sub.empty:
                continue
            color = METHOD_COLORS.get(method, "grey")
            ax.plot(
                sub["alpha"], sub[metric],
                marker="o", color=color, linewidth=2,
                label=METHOD_LABELS.get(method, method), zorder=4,
            )
            # overlay individual S_i dots on the hpo_mean panel
            if metric == "hpo_mean":
                for a in alphas:
                    r = all_results[a].get(method)
                    if r is None:
                        continue
                    scores = r["incumbent_scores"]
                    ax.scatter(
                        [a] * len(scores), scores,
                        color=color, s=12, alpha=0.25, zorder=3,
                    )

        ax.set_xlabel("α", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(alphas)
        ax.legend(fontsize=8, loc="best", framealpha=0.9)

    fig.tight_layout()
    _save(fig, output_dir, "metrics_vs_alpha")
    plt.close(fig)


def plot_incumbents_boxplot(
    all_results: dict[float, dict[str, dict]],
    methods: list[str],
    env_config: str,
    algorithm: str,
    output_dir: Path,
) -> None:
    """Grouped boxplots of per-run incumbent scores S_i, grouped by α."""
    alphas    = sorted(all_results.keys())
    n_methods = len(methods)
    group_w   = 0.75
    box_w     = group_w / n_methods
    offsets   = np.linspace(
        -(group_w - box_w) / 2,
         (group_w - box_w) / 2,
        n_methods,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#f7f7f7")
    ax.grid(axis="y", color="white", linewidth=1.0, zorder=0)
    ax.set_title(
        f"Per-run incumbent transfer scores S_i vs α\n"
        f"{algorithm.upper()} / {env_config}  "
        f"(each box = N independent BO runs, horizontal line = median)",
        fontsize=12, fontweight="bold",
    )

    legend_handles = []
    for m_idx, method in enumerate(methods):
        color = METHOD_COLORS.get(method, "grey")
        positions, data = [], []
        for a_idx, alpha in enumerate(alphas):
            r = all_results[alpha].get(method)
            scores = np.asarray(r["incumbent_scores"]) if r is not None else np.array([])
            positions.append(a_idx + offsets[m_idx])
            data.append(scores)

        ax.boxplot(
            data,
            positions=positions,
            widths=box_w * 0.85,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.55, linewidth=1.2),
            medianprops=dict(color="black", linewidth=2.0),
            whiskerprops=dict(color=color, linewidth=1.2),
            capprops=dict(color=color, linewidth=1.2),
            flierprops=dict(marker="o", color=color, alpha=0.45, markersize=4),
            zorder=3,
        )
        legend_handles.append(
            mlines.Line2D([], [], color=color, linewidth=6, alpha=0.6,
                          label=METHOD_LABELS.get(method, method))
        )

    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"α = {a:g}" for a in alphas], fontsize=11)
    ax.set_xlabel("Risk-aversion α", fontsize=12)
    ax.set_ylabel("S_i  (mean reward over target domains × seeds)", fontsize=11)
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    _save(fig, output_dir, "boxplot_incumbent_scores")
    plt.close(fig)


def plot_incumbent_scores_strip(
    all_results: dict[float, dict[str, dict]],
    methods: list[str],
    env_config: str,
    algorithm: str,
    output_dir: Path,
) -> None:
    """Strip chart (jittered dots + mean bar) of S_i per α, one panel per method."""
    alphas = sorted(all_results.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(4.5 * len(methods), 5), sharey=True)
    if len(methods) == 1:
        axes = [axes]

    fig.suptitle(
        f"Per-run incumbent transfer scores  —  {algorithm.upper()} / {env_config}\n"
        "(dots = individual BO runs, bar = mean)",
        fontsize=12, fontweight="bold",
    )
    rng = np.random.default_rng(0)

    for ax, method in zip(axes, methods):
        color = METHOD_COLORS.get(method, "grey")
        ax.set_facecolor("#f7f7f7")
        ax.grid(axis="y", color="white", linewidth=1.0, zorder=0)

        for a_idx, alpha in enumerate(alphas):
            r = all_results[alpha].get(method)
            if r is None:
                continue
            scores = np.asarray(r["incumbent_scores"])
            jitter = rng.uniform(-0.12, 0.12, len(scores))
            ax.scatter(a_idx + jitter, scores,
                       color=color, s=30, alpha=0.55, edgecolors="none", zorder=4)
            ax.hlines(scores.mean(), a_idx - 0.25, a_idx + 0.25,
                      color=color, linewidth=2.5, zorder=5)

        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"α={a:g}" for a in alphas], fontsize=10)
        ax.set_title(METHOD_LABELS.get(method, method),
                     fontsize=12, fontweight="bold", color=color)
        ax.set_xlabel("α", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("S_i  (transfer score)", fontsize=11)

    fig.tight_layout()
    _save(fig, output_dir, "strip_incumbent_scores")
    plt.close(fig)


def print_summary_table(
    all_results: dict[float, dict[str, dict]],
    methods: list[str],
) -> None:
    col_w = 14
    for alpha in sorted(all_results.keys()):
        print(f"\n  ── α = {alpha:g} ──────────────────────────────────────────────")
        header = f"  {'Metric':<32}" + "".join(
            f"{METHOD_LABELS.get(m, m):>{col_w}}" for m in methods
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for key, label in [
            ("hpo_mean",             "E[S_i]  mean transfer perf."),
            ("hpo_std",              "Std[S_i]  HPO instability"),
            ("hpo_var",              "Var[S_i]  HPO variance"),
            ("hpo_worst_case_score", f"E−α·Std  worst-case (α={alpha:g})"),
            ("hpo_mean_var_score",   f"E−α·Var  mean-var   (α={alpha:g})"),
            ("hpo_min",              "Min S_i"),
            ("hpo_max",              "Max S_i"),
        ]:
            row = f"  {label:<32}"
            for m in methods:
                r = all_results[alpha].get(m)
                val = r[key] if r else float("nan")
                row += f"{val:>{col_w}.4f}"
            print(row)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Sweep α and compare HPO reliability of GPUCB / RAHBO / ERAHBO "
            "after transferring incumbents to held-out target domains."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-root",      default=DEFAULT_ROOT,
                   help="Path to the results_dr directory (dr-dataset_a* subdirs).")
    p.add_argument("--algorithm",         default=DEFAULT_ALGO, choices=["ppo", "dqn", "sac"])
    p.add_argument("--env-config",        default=DEFAULT_ENV)
    p.add_argument("--alphas",            nargs="+", type=float, default=DEFAULT_ALPHAS)
    p.add_argument("--methods",           nargs="+", default=["gpucb", "rahbo", "erahbo"])
    p.add_argument("--n-targets",         type=int,   default=10,
                   help="Number of held-out target domains.")
    p.add_argument("--n-seeds",           type=int,   default=5,
                   help="Training seeds per target domain.")
    p.add_argument("--seed-offset",       type=int,   default=0)
    p.add_argument("--target-seed",       type=int,   default=42,
                   help="RNG seed for sampling target domains (fixed across all α).")
    p.add_argument("--n-total-timesteps", type=float, default=None)
    p.add_argument("--n-eval-steps",      type=int,   default=None)
    p.add_argument("--n-eval-episodes",   type=int,   default=None)
    p.add_argument("--output-dir",        default=DEFAULT_OUTPUT)
    p.add_argument("--no-save",           action="store_true",
                   help="Do not save per-alpha run results (still saves plots).")
    p.add_argument("--load-results",      default=None, metavar="JSON",
                   help="Skip evaluation and just regenerate plots from a saved "
                        "all_results.json file.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.results_root)
    out  = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Short-circuit: re-plot only ───────────────────────────────────────────
    if args.load_results:
        print(f"Loading saved results from {args.load_results} ...")
        all_results = load_all_results(Path(args.load_results))
        methods = list(next(iter(all_results.values())).keys())
    else:
        # ── 1. Env config ─────────────────────────────────────────────────────
        env_cfg   = load_env_config(args.env_config)
        dr_config = env_cfg.get("domain_randomization", {})
        if not dr_config:
            print(
                f"Warning: no 'domain_randomization' section in '{args.env_config}'.",
                file=sys.stderr,
            )

        # ── 2. Fixed shared target domains ────────────────────────────────────
        seed_list = list(range(args.seed_offset, args.seed_offset + args.n_seeds))
        print(f"Sampling {args.n_targets} shared target domains (seed={args.target_seed}) ...")
        target_domains = sample_target_domains(
            dr_config, args.n_targets, rng_seed=args.target_seed
        )
        for i, td in enumerate(target_domains):
            print("  Domain {:>2d}: {}".format(
                i, ", ".join(f"{k}={v:.4f}" for k, v in td.items())
            ))

        methods = args.methods
        print(f"\nMethods   : {methods}")
        print(f"Alphas    : {args.alphas}")
        print(f"Seeds     : {seed_list}")
        print(f"Output    : {out}\n")

        # ── 3. Evaluate each α ────────────────────────────────────────────────
        all_results: dict[float, dict[str, dict]] = {}

        for alpha in args.alphas:
            print(f"\n{'='*72}")
            print(f"  α = {alpha:g}  (RAHBO: mean−α·Var, ERAHBO: mean−α·Std)")
            print(f"{'='*72}")
            all_results[alpha] = evaluate_one_alpha(
                root=root,
                alpha=alpha,
                methods=methods,
                env_cfg=env_cfg,
                algorithm=args.algorithm,
                env_config=args.env_config,
                target_domains=target_domains,
                seed_list=seed_list,
                n_total_timesteps=args.n_total_timesteps,
                n_eval_steps=args.n_eval_steps,
                n_eval_episodes=args.n_eval_episodes,
                output_dir=out if not args.no_save else None,
            )

        # ── 4. Save aggregated results ────────────────────────────────────────
        if not args.no_save:
            save_all_results(all_results, out / "all_results.json")

    # ── 5. Summary table ──────────────────────────────────────────────────────
    print_summary_table(all_results, methods)

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_metrics_vs_alpha(all_results, methods, args.env_config, args.algorithm, out)
    plot_incumbents_boxplot(all_results, methods, args.env_config, args.algorithm, out)
    plot_incumbent_scores_strip(all_results, methods, args.env_config, args.algorithm, out)

    print(f"\nDone.  Outputs in: {out}/")


if __name__ == "__main__":
    main()
