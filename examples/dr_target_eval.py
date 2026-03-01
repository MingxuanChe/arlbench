"""
Domain Randomization Target Evaluation Pipeline

Evaluates a given HP configuration (or samples one from the search space) across
multiple target domains. Target domains are independently sampled from the same DR
distribution defined in the environment config, so they represent the range of
environments the policy may face at deployment time.

For each target domain we train the agent for n_seeds independent seeds, which
gives a distribution of returns that captures both training stochasticity and
environment variability.

Usage examples
--------------
# Use default HP and sample 10 targets x 50 seeds
python examples/dr_target_eval.py --env-config cc_cartpole_dr --algorithm dqn

# Provide a HP config YAML
python examples/dr_target_eval.py \\
    --env-config cc_pendulum_dr --algorithm ppo \\
    --hp-config examples/configs/algorithm/ppo.yaml \\
    --n-targets 15 --n-seeds 30

# Custom budget / output directory
python examples/dr_target_eval.py \\
    --env-config cc_cartpole_dr --algorithm dqn \\
    --n-targets 10 --n-seeds 50 \\
    --n-total-timesteps 1e5 \\
    --output-dir results/dr_eval/my_experiment
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import jax
import numpy as np
import pandas as pd
import yaml

from arlbench.autorl.autorl_env import DEFAULT_AUTO_RL_CONFIG

# ─── paths ────────────────────────────────────────────────────────────────────
EXAMPLES_DIR = Path(__file__).resolve().parent
CONFIGS_ENV_DIR = EXAMPLES_DIR / "configs" / "environment"
CONFIGS_ALG_DIR = EXAMPLES_DIR / "configs" / "algorithm"

# ─── helpers ──────────────────────────────────────────────────────────────────


class _NumpyEncoder(json.JSONEncoder):
    """Convert numpy scalars to native Python types for JSON serialisation."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_env_config(env_config: str) -> dict:
    """Load environment config YAML.

    Accepts either:
    - A stem name like ``cc_cartpole_dr`` (resolved relative to examples/configs/environment/)
    - An absolute / relative path to a YAML file
    """
    path = Path(env_config)
    if not path.suffix:
        path = CONFIGS_ENV_DIR / f"{env_config}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Environment config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_hp_config(hp_config_path: str) -> dict:
    """Load a HP config from a YAML file.

    The file should contain a ``hp_config`` mapping (as in the algorithm YAML
    files shipped with ARLBench).  If ``hp_config`` is absent, the whole
    document is treated as a flat dict of hyperparameters.
    """
    with open(hp_config_path) as f:
        doc = yaml.safe_load(f)
    if isinstance(doc, dict) and "hp_config" in doc:
        return doc["hp_config"]
    return doc


def sample_hp_config(algorithm: str, seed: int = 0) -> dict:
    """Sample a random HP config from the algorithm's ConfigSpace."""
    from arlbench.core.algorithms import DQN, PPO, SAC

    alg_map = {"dqn": DQN, "ppo": PPO, "sac": SAC}
    alg_cls = alg_map[algorithm.lower()]
    cs = alg_cls.get_hpo_config_space()
    cs.seed(seed)
    cfg = cs.sample_configuration()
    return dict(cfg)


def sample_target_domains(
    dr_config: dict,
    n_targets: int,
    rng_seed: int = 0,
) -> list[dict[str, float]]:
    """Sample ``n_targets`` independent target domains from the DR distribution.

    Each domain is a dict mapping parameter name → scalar float value.

    Parameters
    ----------
    dr_config:
        The ``domain_randomization`` sub-dict from the environment config YAML.
    n_targets:
        Number of target domains to sample.
    rng_seed:
        Seed for reproducible domain sampling.

    Returns
    -------
    List of dicts, one per target domain.
    """
    from arlbench.core.environments.domain_randomization import DomainRandomizationConfig

    rng = jax.random.PRNGKey(rng_seed)
    domains: list[dict[str, float]] = []

    for i in range(n_targets):
        rng, split_rng = jax.random.split(rng)
        param_rngs = jax.random.split(split_rng, len(dr_config))

        domain_params: dict[str, float] = {}
        for (param_name, param_cfg), param_rng in zip(dr_config.items(), param_rngs):
            cfg_dict = dict(param_cfg)
            cfg_dict["param_name"] = param_name
            dr_cfg = DomainRandomizationConfig.from_dict(cfg_dict)
            value = dr_cfg.sample(param_rng)
            domain_params[param_name] = float(value)

        domains.append(domain_params)

    return domains


def build_autorl_config(
    env_cfg: dict,
    algorithm: str,
    env_params: dict[str, float],
    seed_list: list[int],
    n_total_timesteps: float | None,
    n_eval_steps: int | None,
    n_eval_episodes: int | None,
    output_dir: Path | None = None,
) -> dict:
    """Build the AutoRLEnv config dict for a single fixed-param target domain.

    Training budget values (n_total_timesteps, n_eval_steps, n_eval_episodes)
    fall back to whatever is specified in the *source-domain* environment config
    YAML so that target-domain evaluations use identical training parameters.

    Note: ``seed`` is set to the first element of ``seed_list`` for init;
    the full list is passed to ``env.step(..., seed=seed_list)`` at call time.
    """
    cfg = {
        "seed": seed_list[0],  # single int for AutoRLEnv init; full list goes to env.step()
        "env_framework": env_cfg["framework"],
        "env_name": env_cfg["name"],
        "env_kwargs": env_cfg.get("kwargs", {}),
        "eval_env_kwargs": env_cfg.get("eval_kwargs", {}),
        # Fixed env params for this target domain – no domain randomization.
        # Both training AND eval env must receive the same fixed params.
        # The training env uses "env_params"; the eval env uses "env_eval_params"
        # (see AutoRLEnv.__init__). The DR copy-over fallback only fires when
        # domain_randomization is non-empty, so we must set both explicitly.
        "env_params": env_params,
        "env_eval_params": env_params,
        "domain_randomization": {},
        "n_envs": env_cfg.get("n_envs", 8),
        "algorithm": algorithm.lower(),
        "cnn_policy": env_cfg.get("cnn_policy", False),
        "deterministic_eval": env_cfg.get("deterministic_eval", True),
        "nas_config": {},
        # Read budget from CLI if given, otherwise match source-domain env config,
        # falling back to the ARLBench framework defaults (DEFAULT_AUTO_RL_CONFIG).
        # PyYAML parses scientific notation (e.g. 1e5) as a string, so coerce to int.
        "n_total_timesteps": int(float(
            n_total_timesteps
            or env_cfg.get("n_total_timesteps", DEFAULT_AUTO_RL_CONFIG["n_total_timesteps"])
        )),
        "checkpoint": [],
        "checkpoint_name": "default_checkpoint",
        "checkpoint_dir": str(output_dir / "checkpoints") if output_dir else "checkpoints",
        "state_features": [],
        "objectives": ["reward_mean"],
        "optimize_objectives": "upper",
        "n_eval_steps": (
            n_eval_steps
            if n_eval_steps is not None
            else env_cfg.get("n_eval_steps", DEFAULT_AUTO_RL_CONFIG["n_eval_steps"])
        ),
        "n_eval_episodes": (
            n_eval_episodes
            if n_eval_episodes is not None
            else env_cfg.get("n_eval_episodes", DEFAULT_AUTO_RL_CONFIG["n_eval_episodes"])
        ),
        "n_steps": 1,
    }
    return cfg


def run_target_domain(
    env_cfg: dict,
    algorithm: str,
    hp_config: dict,
    env_params: dict[str, float],
    seed_list: list[int],
    n_total_timesteps: float | None,
    n_eval_steps: int | None,
    n_eval_episodes: int | None,
    output_dir: Path | None = None,
) -> np.ndarray:
    """Train + evaluate one fixed target domain for all seeds.

    Returns
    -------
    rewards: np.ndarray, shape (n_seeds,)
        Final-checkpoint mean episode return for each seed.
    """
    from arlbench.autorl import AutoRLEnv

    cfg = build_autorl_config(
        env_cfg,
        algorithm,
        env_params,
        seed_list,
        n_total_timesteps,
        n_eval_steps,
        n_eval_episodes,
        output_dir=output_dir,
    )

    env = AutoRLEnv(config=cfg)
    env.reset()
    _, objectives, _, _, _ = env.step(hp_config, seed=seed_list)  # list → multi-seed vmap

    rewards = np.asarray(objectives["reward_mean"])  # shape: (n_seeds,)
    return rewards


def aggregate_results(
    all_rewards: np.ndarray,
    alpha: float = 1.0,
    beta: float = 2.5,
    cvar_quantile: float = 0.1,
) -> dict:
    """Compute aggregate statistics over all (target, seed) evaluations.

    All three headline metrics are computed over the *per-domain means* so that
    each domain counts equally regardless of the number of seeds.

    Parameters
    ----------
    all_rewards:
        Array of shape (n_targets, n_seeds).
    alpha:
        Variance penalty for the mean-variance score
        ``mean - alpha * variance``.  Default 1.0.
    beta:
        Standard-deviation penalty for the worst-case score
        ``mean - beta * std``.  Default 2.5.
    cvar_quantile:
        Quantile for CVaR (conditional value-at-risk / expected shortfall).
        Default 0.1 corresponds to the bottom 10%.

    Returns
    -------
    Dictionary of aggregate metrics.
    """
    per_domain_means = all_rewards.mean(axis=1)   # (n_targets,)
    per_domain_stds  = all_rewards.std(axis=1)    # (n_targets,)

    mean_of_means = float(per_domain_means.mean())
    std_of_means  = float(per_domain_means.std())
    var_of_means  = float(per_domain_means.var())

    # ── three headline metrics ────────────────────────────────────────────────
    # 1. Mean performance (with std)
    mean_perf        = mean_of_means
    mean_perf_std    = std_of_means

    # 2. Mean-variance score:  mean - alpha * variance  (higher is better)
    mean_var_score   = mean_of_means - alpha * var_of_means

    # 3. Worst-case score:  mean - beta * std  (lower-confidence-bound)
    worst_case_score = mean_of_means - beta * std_of_means

    # ── additional diagnostics ────────────────────────────────────────────────
    flat = all_rewards.flatten()
    sorted_means = np.sort(per_domain_means)
    n_tail = max(1, int(np.ceil(cvar_quantile * len(sorted_means))))
    cvar   = float(sorted_means[:n_tail].mean())

    return {
        "n_targets": int(all_rewards.shape[0]),
        "n_seeds": int(all_rewards.shape[1]),
        # Headline metrics
        "mean_performance":       mean_perf,
        "mean_performance_std":   mean_perf_std,
        "mean_variance_score":    mean_var_score,
        "alpha":                  alpha,
        "worst_case_score":       worst_case_score,
        "beta":                   beta,
        # Diagnostics
        "per_domain_min_mean":    float(per_domain_means.min()),
        "per_domain_max_mean":    float(per_domain_means.max()),
        "per_domain_median_mean": float(np.median(per_domain_means)),
        "global_min":             float(flat.min()),
        "global_max":             float(flat.max()),
        f"cvar_{int(cvar_quantile * 100)}pct": cvar,
    }


def print_summary(
    summary: dict,
    hp_config: dict,
    target_domains: list[dict],
    all_rewards: np.ndarray,
) -> None:
    """Human-readable summary to stdout."""
    sep = "=" * 72

    print(sep)
    print("  Domain Randomization Target Evaluation – Summary")
    print(sep)

    print("\n[HP Configuration]")
    for k, v in hp_config.items():
        print(f"  {k}: {v}")

    print("\n[Evaluation Setup]")
    print(f"  Number of target domains : {summary['n_targets']}")
    print(f"  Seeds per domain         : {summary['n_seeds']}")
    print(f"  Total evaluations        : {summary['n_targets'] * summary['n_seeds']}")

    alpha = summary["alpha"]
    beta  = summary["beta"]

    print("\n[Headline Metrics]  (computed over per-domain means)")
    print(f"  Mean performance         : {summary['mean_performance']:>10.4f}"
          f"  ±  {summary['mean_performance_std']:.4f}")
    print(f"  Mean-variance score      : {summary['mean_variance_score']:>10.4f}"
          f"  (mean - {alpha}·variance)")
    print(f"  Worst-case score         : {summary['worst_case_score']:>10.4f}"
          f"  (mean - {beta}·std)")

    print("\n[Diagnostics]")
    print(f"  Per-domain min mean      : {summary['per_domain_min_mean']:>10.4f}")
    print(f"  Per-domain median mean   : {summary['per_domain_median_mean']:>10.4f}")
    print(f"  Per-domain max mean      : {summary['per_domain_max_mean']:>10.4f}")
    cvar_key = [k for k in summary if k.startswith("cvar")][0]
    print(f"  {cvar_key.upper():<27}: {summary[cvar_key]:>10.4f}")
    print(f"  Global min               : {summary['global_min']:>10.4f}")
    print(f"  Global max               : {summary['global_max']:>10.4f}")

    print("\n[Per-Domain Breakdown]")
    header = f"  {'Domain':>8}  {'Mean':>10}  {'Std':>10}  {'Min':>10}  {'Max':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, row in enumerate(all_rewards):
        print(
            f"  {i:>8d}  {row.mean():>10.4f}  {row.std():>10.4f}"
            f"  {row.min():>10.4f}  {row.max():>10.4f}"
        )
    print(sep)


def save_results(
    output_dir: Path,
    target_domains: list[dict],
    all_rewards: np.ndarray,
    hp_config: dict,
    summary: dict,
    seed_list: list[int],
    env_cfg: dict,
    algorithm: str,
) -> None:
    """Persist all outputs to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Target domain parameters
    domains_path = output_dir / "target_domains.json"
    with open(domains_path, "w") as f:
        json.dump(target_domains, f, indent=2)
    print(f"\nSaved target domains  → {domains_path}")

    # 2. HP config
    hp_path = output_dir / "hp_config.yaml"
    with open(hp_path, "w") as f:
        yaml.dump(hp_config, f, default_flow_style=False)
    print(f"Saved HP config       → {hp_path}")

    # 3. Raw results CSV: one row per (domain_idx, seed)
    rows = []
    for d_idx, (params, rewards) in enumerate(zip(target_domains, all_rewards)):
        for s_idx, (seed, reward) in enumerate(zip(seed_list, rewards)):
            row = {"domain_idx": d_idx, "seed": seed, "reward": float(reward)}
            row.update({f"param_{k}": v for k, v in params.items()})
            rows.append(row)
    results_df = pd.DataFrame(rows)
    results_path = output_dir / "results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved raw results     → {results_path}")

    # 4. Per-domain summary CSV
    domain_rows = []
    for d_idx, (params, rewards) in enumerate(zip(target_domains, all_rewards)):
        row = {
            "domain_idx": d_idx,
            "mean": float(rewards.mean()),
            "std": float(rewards.std()),
            "min": float(rewards.min()),
            "max": float(rewards.max()),
        }
        row.update({f"param_{k}": v for k, v in params.items()})
        domain_rows.append(row)
    domain_df = pd.DataFrame(domain_rows)
    domain_path = output_dir / "per_domain_summary.csv"
    domain_df.to_csv(domain_path, index=False)
    print(f"Saved domain summary  → {domain_path}")

    # 5. Aggregate summary JSON
    metadata = {
        "env_name": env_cfg.get("name"),
        "algorithm": algorithm,
        "n_envs": env_cfg.get("n_envs", 8),
        "hp_config": hp_config,
        "seeds": seed_list,
    }
    summary_out = {"metadata": metadata, "aggregate": summary}
    summary_path = output_dir / "summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary_out, f, indent=2, cls=_NumpyEncoder)
    print(f"Saved summary         → {summary_path}")


# ─── main ─────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a HP config across DR target domains.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Environment / algorithm
    parser.add_argument(
        "--env-config",
        default="cc_cartpole_dr",
        help=(
            "Environment config name (resolved from examples/configs/environment/) "
            "or path to a YAML file."
        ),
    )
    parser.add_argument(
        "--algorithm",
        default="dqn",
        choices=["dqn", "ppo", "sac"],
        help="RL algorithm to use.",
    )

    # HP configuration
    parser.add_argument(
        "--hp-config",
        default=None,
        help=(
            "Path to a YAML file containing the HP config (with an ``hp_config`` key "
            "or as a flat dict). If omitted, a configuration is sampled from the "
            "algorithm's default search space."
        ),
    )
    parser.add_argument(
        "--hp-sample-seed",
        type=int,
        default=0,
        help="Seed used when sampling a random HP config (only used if --hp-config is omitted).",
    )

    # Evaluation scale
    parser.add_argument(
        "--n-targets",
        type=int,
        default=10,
        help="Number of target domains to sample from the DR distribution.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=50,
        help="Number of training seeds per target domain.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=42,
        help="Seeds will be [seed_offset, seed_offset+1, ..., seed_offset+n_seeds-1].",
    )
    parser.add_argument(
        "--target-seed",
        type=int,
        default=0,
        help="RNG seed for sampling target domain parameters.",
    )

    # Training budget (overrides environment config values when given)
    parser.add_argument(
        "--n-total-timesteps",
        type=float,
        default=None,
        help="Total training timesteps. Defaults to the value in the environment config.",
    )
    parser.add_argument(
        "--n-eval-steps",
        type=int,
        default=None,
        help=(
            "Number of evaluation checkpoints during training. "
            "Defaults to the value in the environment config (field 'n_eval_steps'), "
            "or the ARLBench framework default (DEFAULT_AUTO_RL_CONFIG['n_eval_steps']) if absent."
        ),
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=None,
        help=(
            "Evaluation episodes per checkpoint. "
            "Defaults to the value in the environment config (field 'n_eval_episodes'), "
            "or the ARLBench framework default (DEFAULT_AUTO_RL_CONFIG['n_eval_episodes']) if absent."
        ),
    )

    # Risk-sensitivity weights
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Variance penalty weight for mean-variance score: mean - alpha*variance.  Default 1.0.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=2.5,
        help="Std penalty weight for worst-case score: mean - beta*std.  Default 2.5.",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to save results. Defaults to "
            "results/dr_eval/<algorithm>_<env_name>."
        ),
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to disk (useful for quick tests).",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ── 1. Environment config ──────────────────────────────────────────────────
    print(f"Loading environment config: {args.env_config}")
    env_cfg = load_env_config(args.env_config)

    dr_config = env_cfg.get("domain_randomization", {})
    if not dr_config:
        raise ValueError(
            f"Environment config '{args.env_config}' does not contain a "
            "'domain_randomization' section.  This pipeline requires DR to be defined."
        )

    print(f"  Environment : {env_cfg['name']} ({env_cfg['framework']})")
    print(f"  DR params   : {list(dr_config.keys())}")

    # ── 2. HP configuration ───────────────────────────────────────────────────
    if args.hp_config:
        print(f"\nLoading HP config from: {args.hp_config}")
        hp_config = load_hp_config(args.hp_config)
    else:
        print(f"\nSampling HP config from {args.algorithm.upper()} search space "
              f"(seed={args.hp_sample_seed}) ...")
        hp_config = sample_hp_config(args.algorithm, seed=args.hp_sample_seed)

    print("  HP config:")
    for k, v in hp_config.items():
        print(f"    {k}: {v}")

    # ── 3. Target domains ─────────────────────────────────────────────────────
    print(f"\nSampling {args.n_targets} target domains (seed={args.target_seed}) ...")
    target_domains = sample_target_domains(dr_config, args.n_targets, rng_seed=args.target_seed)
    for i, td in enumerate(target_domains):
        params_str = ", ".join(f"{k}={v:.4f}" for k, v in td.items())
        print(f"  Domain {i:>3d}: {params_str}")

    # ── 4. Seed list ──────────────────────────────────────────────────────────
    seed_list = list(range(args.seed_offset, args.seed_offset + args.n_seeds))

    # ── 5. Output directory (needed early so checkpoint_dir avoids /tmp) ──────
    if not args.no_save:
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            env_name_slug = env_cfg["name"].replace("-", "_").lower()
            output_dir = Path("results") / "dr_eval" / f"{args.algorithm}_{env_name_slug}"
    else:
        output_dir = None

    # ── 6. Evaluate across all target domains ─────────────────────────────────
    all_rewards: list[np.ndarray] = []
    n_targets = len(target_domains)

    print(f"\nRunning {n_targets} target domains × {args.n_seeds} seeds "
          f"= {n_targets * args.n_seeds} total evaluations ...\n")

    for i, domain_params in enumerate(target_domains):
        params_str = ", ".join(f"{k}={v:.4f}" for k, v in domain_params.items())
        print(f"[{i+1}/{n_targets}] Target domain: {params_str}")

        rewards = run_target_domain(
            env_cfg=env_cfg,
            algorithm=args.algorithm,
            hp_config=hp_config,
            env_params=domain_params,
            seed_list=seed_list,
            n_total_timesteps=args.n_total_timesteps,
            n_eval_steps=args.n_eval_steps,
            n_eval_episodes=args.n_eval_episodes,
            output_dir=output_dir if not args.no_save else None,
        )

        all_rewards.append(rewards)
        domain_mean = float(rewards.mean())
        domain_std = float(rewards.std())
        print(f"         → mean={domain_mean:.4f}  std={domain_std:.4f}")

    all_rewards_arr = np.stack(all_rewards)  # (n_targets, n_seeds)

    # ── 7. Aggregate & print ──────────────────────────────────────────────────
    summary = aggregate_results(all_rewards_arr, alpha=args.alpha, beta=args.beta)
    print_summary(summary, hp_config, target_domains, all_rewards_arr)

    # ── 8. Save ───────────────────────────────────────────────────────────────
    if not args.no_save:
        save_results(
            output_dir=output_dir,
            target_domains=target_domains,
            all_rewards=all_rewards_arr,
            hp_config=hp_config,
            summary=summary,
            seed_list=seed_list,
            env_cfg=env_cfg,
            algorithm=args.algorithm,
        )


if __name__ == "__main__":
    main()
