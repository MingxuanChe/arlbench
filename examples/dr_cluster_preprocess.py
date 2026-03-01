"""
dr_cluster_preprocess.py
========================
Stage 0 of the cluster evaluation pipeline.

Must be run **once on the login node** before submitting the SLURM array job.

What it does
------------
1. Discovers all (alpha, method) run directories under ``--results-root``.
2. Extracts the principled BO incumbent (from ``results.pkl``) for every run
   and writes each one as a YAML file to ``<output-dir>/incumbents/``.
3. Samples a single shared set of target domains and writes
   ``<output-dir>/target_domains.json``.
4. Writes ``<output-dir>/eval_manifest.json`` — one entry per (alpha, method,
   run_idx) triple.  The SLURM array task ID indexes directly into this list.
5. Prints the total number of tasks so the ``--array`` range is clear.

Usage
-----
pixi run python examples/dr_cluster_preprocess.py \\
    --results-root /home/mingxuan/Repos/Efficient-Risk-Averse-BO/results_dr \\
    --algorithm ppo --env-config cc_acrobot_dr \\
    --alphas 0.5 1.0 1.5 2.0 \\
    --n-targets 20 --target-seed 0 \\
    --output-dir results/cluster_eval/ppo_acrobot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── make sure examples/ is on the path ───────────────────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dr_compare_methods import (  # noqa: E402
    _extract_incumbents_from_pkl,
    _hp_names_from_trace,
    _load_pkl_data,
    sanitize_hp_config,
    _decode_pkl_incumbent,
)
from dr_target_eval import load_env_config, sample_target_domains  # noqa: E402

# ─── path resolution (mirrors plot_alpha_sweep.py) ───────────────────────────

def _run_dir(root: Path, alpha: float, method: str, algorithm: str, env_config: str) -> Path:
    """Return the directory holding ``results.pkl`` for a given combo."""
    alpha_dir = root / f"dr-dataset_a{alpha:.1f}"
    base = alpha_dir / method / "arlbench_dataset" / algorithm / env_config

    if method == "gpucb":
        return base

    # rahbo / erahbo have an alpha sub-folder
    for sub in (f"{alpha:.1f}", f"{alpha:g}", str(int(alpha))):
        candidate = base / sub
        if candidate.is_dir():
            return candidate

    return base / f"{alpha:.1f}"   # will raise clearly downstream if missing


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 0: extract incumbents and build eval manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-root", default="/home/mingxuan/Repos/Efficient-Risk-Averse-BO/results_dr")
    parser.add_argument("--algorithm",   required=True, help="e.g. ppo / sac / dqn")
    parser.add_argument("--env-config",  required=True, help="e.g. cc_acrobot_dr")
    parser.add_argument("--methods",     nargs="+", default=["gpucb", "rahbo", "erahbo"])
    parser.add_argument("--alphas",      nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--n-targets",   type=int,   default=20)
    parser.add_argument("--target-seed", type=int,   default=0,
                        help="RNG seed for target domain sampling (must match aggregate stage).")
    parser.add_argument("--output-dir",  required=True)
    args = parser.parse_args()

    root       = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inc_dir = output_dir / "incumbents"
    inc_dir.mkdir(exist_ok=True)

    # ── 1. Sample shared target domains ──────────────────────────────────────
    env_cfg   = load_env_config(args.env_config)
    dr_config = env_cfg.get("domain_randomization", {})
    target_domains = sample_target_domains(dr_config, args.n_targets, rng_seed=args.target_seed)

    td_path = output_dir / "target_domains.json"
    with open(td_path, "w") as f:
        json.dump(target_domains, f, indent=2)
    print(f"Sampled {len(target_domains)} target domains → {td_path}")

    # ── 2. Extract incumbents → per-run YAML files + build manifest ───────────
    manifest: list[dict] = []
    task_id = 0

    for alpha in args.alphas:
        for method in args.methods:
            run_dir = _run_dir(root, alpha, method, args.algorithm, args.env_config)
            pkl_path = run_dir / "results.pkl"

            if not pkl_path.exists():
                print(f"  [SKIP] results.pkl not found: {pkl_path}")
                continue

            print(f"\n[α={alpha:g}  {method}]  {run_dir}")

            # Decode incumbents (caches to incumbent_reporting.json next to pkl)
            try:
                incumbents, source_scores, source_score_vars = _extract_incumbents_from_pkl(
                    pkl_path, run_dir, args.algorithm,
                    output_dir=inc_dir / f"a{alpha:g}_{method}",
                )
            except Exception as exc:
                print(f"  [ERROR] {exc}")
                continue

            for run_idx, (hp, src, src_var) in enumerate(zip(incumbents, source_scores, source_score_vars)):
                yaml_path = inc_dir / f"a{alpha:g}_{method}" / "incumbent_from_pkl" / f"run_{run_idx:02d}_incumbent.yaml"
                if not yaml_path.exists():
                    print(f"  [WARN] yaml not written for run {run_idx}")
                    continue

                manifest.append({
                    "task_id":       task_id,
                    "alpha":         alpha,
                    "method":        method,
                    "run_idx":       run_idx,
                    "incumbent_yaml": str(yaml_path),
                    "source_score":     float(src),
                    "source_score_var": float(src_var) if src_var is not None else None,
                })
                task_id += 1

            print(f"  → {len(incumbents)} incumbents extracted")

    # ── 3. Write manifest ─────────────────────────────────────────────────────
    manifest_path = output_dir / "eval_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Total eval tasks : {task_id}")
    print(f"  Manifest         : {manifest_path}")
    print(f"  Target domains   : {td_path}")
    print(f"{'='*60}")
    print(f"\nSLURM array flag:  --array=0-{task_id - 1}")


if __name__ == "__main__":
    main()
