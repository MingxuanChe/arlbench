"""
dr_cluster_eval.py
==================
Stage 1 of the cluster evaluation pipeline — runs as one SLURM array task.

Each task evaluates a **single incumbent** across all shared target domains
and all training seeds, then writes a raw rewards file.

The SLURM_ARRAY_TASK_ID directly indexes ``eval_manifest.json`` which was
produced by ``dr_cluster_preprocess.py``.

Usage (direct / debugging)
--------------------------
pixi run python examples/dr_cluster_eval.py \\
    --task-id 0 \\
    --manifest   results/cluster_eval/ppo_acrobot/eval_manifest.json \\
    --target-domains results/cluster_eval/ppo_acrobot/target_domains.json \\
    --env-config cc_acrobot_dr \\
    --algorithm  ppo \\
    --n-seeds    10 \\
    --output-dir results/cluster_eval/ppo_acrobot

In the SLURM script ``--task-id $SLURM_ARRAY_TASK_ID`` is passed automatically.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from dr_target_eval import load_env_config, run_target_domain  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: evaluate one incumbent on all target domains.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task-id",        type=int, required=True,
                        help="Index into eval_manifest.json (= SLURM_ARRAY_TASK_ID).")
    parser.add_argument("--manifest",        required=True,
                        help="Path to eval_manifest.json produced by preprocess step.")
    parser.add_argument("--target-domains",  required=True,
                        help="Path to target_domains.json produced by preprocess step.")
    parser.add_argument("--env-config",      required=True)
    parser.add_argument("--algorithm",       required=True)
    parser.add_argument("--n-seeds",         type=int, default=10)
    parser.add_argument("--seed-offset",     type=int, default=20)
    parser.add_argument("--n-total-timesteps", type=float, default=None)
    parser.add_argument("--n-eval-steps",    type=int, default=None)
    parser.add_argument("--n-eval-episodes", type=int, default=None)
    parser.add_argument("--output-dir",      required=True)
    args = parser.parse_args()

    t0 = time.time()
    output_dir = Path(args.output_dir)

    # ── Load manifest entry ───────────────────────────────────────────────────
    with open(args.manifest) as f:
        manifest = json.load(f)

    if args.task_id >= len(manifest):
        raise IndexError(
            f"task_id={args.task_id} but manifest has only {len(manifest)} entries."
        )
    entry = manifest[args.task_id]
    alpha    = entry["alpha"]
    method   = entry["method"]
    run_idx  = entry["run_idx"]
    src_score     = float(entry.get("source_score",     float("nan")))
    src_score_var = float(entry.get("source_score_var", float("nan")))
    if src_score_var is None:
        src_score_var = float("nan")

    print(f"Task {args.task_id:>4d}  α={alpha:g}  method={method}  run={run_idx:02d}")

    # ── Load HP config ────────────────────────────────────────────────────────
    import yaml  # noqa: PLC0415
    with open(entry["incumbent_yaml"]) as f:
        raw = yaml.safe_load(f)
    hp_config: dict = raw["hp_config"]
    print(f"  HP config: {hp_config}")

    # ── Load shared target domains ────────────────────────────────────────────
    with open(args.target_domains) as f:
        target_domains: list[dict] = json.load(f)
    n_targets = len(target_domains)
    print(f"  Target domains: {n_targets}")

    # ── Load env config ───────────────────────────────────────────────────────
    env_cfg   = load_env_config(args.env_config)
    seed_list = list(range(args.seed_offset, args.seed_offset + args.n_seeds))

    # ── Evaluate across all target domains ───────────────────────────────────
    # all_rewards shape: (n_targets, n_seeds)
    all_rewards = np.zeros((n_targets, args.n_seeds), dtype=float)

    for j, domain_params in enumerate(target_domains):
        params_str = "  ".join(f"{k}={v:.4f}" for k, v in domain_params.items())
        print(f"  domain [{j+1:>3d}/{n_targets}] {params_str}", end="", flush=True)

        inc_out = output_dir / "raw" / f"a{alpha:g}_{method}_run{run_idx:02d}" / f"domain_{j:03d}"
        rewards = run_target_domain(
            env_cfg=env_cfg,
            algorithm=args.algorithm,
            hp_config=hp_config,
            env_params=domain_params,
            seed_list=seed_list,
            n_total_timesteps=args.n_total_timesteps,
            n_eval_steps=args.n_eval_steps,
            n_eval_episodes=args.n_eval_episodes,
            output_dir=inc_out,
        )
        all_rewards[j, :] = rewards
        print(f"  → mean={rewards.mean():.3f}  std={rewards.std():.3f}")

    # ── Save raw rewards ──────────────────────────────────────────────────────
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"task_{args.task_id:04d}.npz"
    np.savez(
        out_path,
        all_rewards  = all_rewards,   # (n_targets, n_seeds)
        alpha        = np.array(alpha),
        run_idx      = np.array(run_idx),
        source_score     = np.array(src_score),
        source_score_var = np.array(src_score_var),
        task_id          = np.array(args.task_id),
    )
    # also save a method/run label as plain text beside the npz
    meta_path = raw_dir / f"task_{args.task_id:04d}.meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "task_id":      args.task_id,
            "alpha":        alpha,
            "method":       method,
            "run_idx":      run_idx,
            "source_score":     src_score,
            "source_score_var": src_score_var,
            "n_targets":    n_targets,
            "n_seeds":      args.n_seeds,
            "incumbent_score": float(all_rewards.mean()),
        }, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Saved → {out_path}")
    print(f"  Incumbent score S_i = {all_rewards.mean():.4f}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
