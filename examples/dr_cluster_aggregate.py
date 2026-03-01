"""
dr_cluster_aggregate.py
=======================
Stage 2 of the cluster evaluation pipeline.

Loads all ``task_*.npz`` + ``task_*.meta.json`` files from
``output_dir/raw/``, computes HPO reliability metrics for every
(alpha, method) combination, and writes

    output_dir/
        comparison_summary_{a<alpha>}.csv   — one row per method (μ, σ, worst-case, …)
        incumbent_scores_{a<alpha>}.csv     — one row per incumbent
        comparison_summary_all.csv          — all alphas stacked

It also calls ``plot_alpha_results.py`` helpers to regenerate the ridge
and scatter plots when matplotlib is available.

Usage
-----
pixi run python examples/dr_cluster_aggregate.py \\
    --output-dir  results/cluster_eval/ppo_acrobot \\
    --env-config  cc_acrobot_dr \\
    --algorithm   ppo \\
    --alphas 0.5 1.0 1.5 2.0
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ─── helpers ────────────────────────────────────────────────────────────────

def load_raw(raw_dir: Path) -> List[dict]:
    """Return list of records, each enriched with all_rewards array."""
    records = []
    for meta_path in sorted(raw_dir.glob("task_*.meta.json")):
        with open(meta_path) as f:
            meta = json.load(f)
        npz_path = raw_dir / f"task_{meta['task_id']:04d}.npz"
        if not npz_path.exists():
            print(f"  WARNING: {npz_path} missing — skipping task {meta['task_id']}")
            continue
        data = np.load(npz_path)
        meta["all_rewards"] = data["all_rewards"]   # (n_targets, n_seeds)
        records.append(meta)
    return records


def compute_incumbent_score(all_rewards: np.ndarray) -> float:
    """Incumbent score S_i = mean over all (target, seed) rewards."""
    return float(np.mean(all_rewards))


def transfer_gap(source: float, target: float) -> float:
    if abs(source) < 1e-9:
        return float("nan")
    return (source - target) / abs(source)


def aggregate_by_alpha_method(
    records: List[dict],
) -> Dict[Tuple[float, str], List[dict]]:
    """Group records by (alpha, method)."""
    groups: Dict[Tuple[float, str], List[dict]] = defaultdict(list)
    for r in records:
        groups[(float(r["alpha"]), r["method"])].append(r)
    return groups


def compute_metrics(group: List[dict]) -> dict:
    """Given all incumbents for one (alpha, method), return summary metrics."""
    scores   = np.array([compute_incumbent_score(r["all_rewards"]) for r in group])
    src      = np.array([float(r["source_score"]) for r in group])
    src_vars = np.array([float(r.get("source_score_var", float("nan"))) for r in group])
    gaps     = np.array([transfer_gap(s, t) for s, t in zip(src, scores)])
    alpha_val = float(group[0]["alpha"])
    target_mean_var = float(scores.mean() - alpha_val * scores.var())
    # Source mean−α·Var per incumbent on source domain (the BO training objective)
    src_mean_var = src - alpha_val * src_vars
    return {
        "n_incumbents":               len(group),
        "target_score_mean":          float(scores.mean()),
        "target_score_std":           float(scores.std()),
        "target_score_min":           float(scores.min()),
        "target_score_max":           float(scores.max()),
        "hpo_worst_case":             float(scores.min()),
        "target_mean_var_score":      target_mean_var,
        "transfer_gap_mean":          float(np.nanmean(gaps)),
        "transfer_gap_std":           float(np.nanstd(gaps)),
        "source_score_mean":          float(src.mean()),
        "source_score_var_mean":      float(np.nanmean(src_vars)),
        "source_mean_var_score_mean": float(np.nanmean(src_mean_var)),  # E[μ−α·Var] on source
    }


# ─── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: aggregate raw npz rewards into metrics + plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir",  required=True)
    parser.add_argument("--env-config",  default=None)
    parser.add_argument("--algorithm",   default=None)
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0],
        help="Alphas to include in the aggregated summary (all present by default).",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_dir    = output_dir / "raw"

    if not raw_dir.exists():
        sys.exit(f"ERROR: {raw_dir} does not exist — run preprocess + eval first.")

    # ── load all records ──────────────────────────────────────────────────────
    records = load_raw(raw_dir)
    if not records:
        sys.exit(f"ERROR: no task_*.meta.json found in {raw_dir}")

    total_tasks = len(records)
    present_alphas = sorted({float(r["alpha"]) for r in records})
    present_methods = sorted({r["method"] for r in records})
    print(f"Loaded {total_tasks} task results.")
    print(f"  Alphas:   {present_alphas}")
    print(f"  Methods:  {present_methods}")

    # ── group and compute metrics ─────────────────────────────────────────────
    groups   = aggregate_by_alpha_method(records)
    all_rows = []   # for stacked CSV

    per_alpha_summary: Dict[float, pd.DataFrame] = {}
    per_alpha_scores:  Dict[float, pd.DataFrame] = {}

    active_alphas = [a for a in args.alphas if a in present_alphas]
    if not active_alphas:
        active_alphas = present_alphas

    for alpha in active_alphas:
        alpha_rows   = []
        score_rows   = []

        for method in present_methods:
            key   = (alpha, method)
            group = groups.get(key, [])
            if not group:
                print(f"  WARNING: no data for α={alpha:g}, method={method} — skipping")
                continue

            metrics = compute_metrics(group)
            row = {"alpha": alpha, "method": method, **metrics}
            alpha_rows.append(row)
            all_rows.append(row)

            # Per-incumbent rows
            scores = [compute_incumbent_score(r["all_rewards"]) for r in group]
            for r, s in zip(group, scores):
                src_var_i = float(r.get("source_score_var", float("nan")))
                score_rows.append({
                    "alpha":             alpha,
                    "method":            method,
                    "run_idx":           r["run_idx"],
                    "source_score":      r["source_score"],
                    "source_score_var":  src_var_i,
                    "source_mean_var_score": r["source_score"] - float(alpha) * src_var_i,
                    "target_score":      s,
                    "transfer_gap":      transfer_gap(r["source_score"], s),
                    "task_id":           r["task_id"],
                })

        df_summary = pd.DataFrame(alpha_rows)
        df_scores  = pd.DataFrame(score_rows)
        per_alpha_summary[alpha] = df_summary
        per_alpha_scores[alpha]  = df_scores

        # Save CSVs per alpha
        summ_path   = output_dir / f"comparison_summary_a{alpha:g}.csv"
        scores_path = output_dir / f"incumbent_scores_a{alpha:g}.csv"
        df_summary.to_csv(summ_path,   index=False)
        df_scores.to_csv(scores_path,  index=False)
        print(f"\n  α={alpha:g}:")
        print(df_summary[["method", "target_score_mean", "target_score_std",
                            "hpo_worst_case", "target_mean_var_score",
                            "transfer_gap_mean"]].to_string(index=False))
        print(f"  Saved → {summ_path}")
        print(f"  Saved → {scores_path}")

    # ── stacked across all alphas ─────────────────────────────────────────────
    all_df = pd.DataFrame(all_rows)
    all_path = output_dir / "comparison_summary_all.csv"
    all_df.to_csv(all_path, index=False)
    print(f"\nAll-alpha summary → {all_path}")

    # ── optional plotting ─────────────────────────────────────────────────────
    if not args.no_plots:
        try:
            import importlib
            spec = importlib.util.spec_from_file_location(
                "plot_alpha_results",
                str(_HERE / "plot_alpha_results.py"),
            )
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[union-attr]

                for alpha in active_alphas:
                    df_scores = per_alpha_scores[alpha]
                    # rebuild the structure expected by plot_alpha_results helpers
                    results: dict = {}
                    for method in df_scores["method"].unique():
                        sub = df_scores[df_scores["method"] == method]
                        results[method] = {
                            "target_scores":  sub["target_score"].to_numpy(),
                            "source_scores":  sub["source_score"].to_numpy(),
                            "transfer_gaps":  sub["transfer_gap"].to_numpy(),
                            "metrics": per_alpha_summary[alpha]
                                        .set_index("method")
                                        .loc[method]
                                        .to_dict(),
                        }

                    plots_dir = output_dir / "plots" / f"a{alpha:g}"
                    plots_dir.mkdir(parents=True, exist_ok=True)

                    if hasattr(mod, "plot_ridge"):
                        fig = mod.plot_ridge(results, alpha=alpha)
                        ridge_path = plots_dir / "ridge.pdf"
                        fig.savefig(ridge_path, bbox_inches="tight")
                        print(f"  Ridge plot → {ridge_path}")

                    if hasattr(mod, "plot_scatter"):
                        fig = mod.plot_scatter(results, alpha=alpha)
                        scatter_path = plots_dir / "scatter.pdf"
                        fig.savefig(scatter_path, bbox_inches="tight")
                        print(f"  Scatter plot → {scatter_path}")
            else:
                print("  (plot_alpha_results.py not found — skipping plots)")
        except Exception as exc:  # noqa: BLE001
            print(f"  Plotting failed: {exc}")


if __name__ == "__main__":
    main()
