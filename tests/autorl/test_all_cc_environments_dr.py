"""
Test script for domain randomization across all classic control environments.

This script tests domain randomization for:
- CartPole-v1
- Pendulum-v1
- Acrobot-v1
- MountainCar-v0
- MountainCarContinuous-v0
"""

import jax

jax.config.update("jax_enable_x64", True)

from arlbench import AutoRLEnv
import numpy as np


def test_environment(env_name, dr_config, n_seeds=3):
    """Test domain randomization for a specific environment."""
    print(f"\n{'='*70}")
    print(f"Testing {env_name}")
    print(f"{'='*70}")

    config = {
        "seed": 42,
        "env_framework": "gymnax",
        "env_name": env_name,
        "n_envs": 4,
        "algorithm": "dqn" if "CartPole" in env_name or "MountainCar" in env_name or "Acrobot" in env_name else "sac",
        "cnn_policy": False,
        "n_total_timesteps": 5e2,  # Very short for testing
        "n_eval_steps": 2,
        "checkpoint": [],
        "objectives": ["reward_mean"],
        "state_features": [],
        "n_steps": 1,
        "domain_randomization": dr_config,
        "eval_domain_randomization": dr_config,
    }

    print(f"Creating environment with domain randomization...")
    env = AutoRLEnv(config=config)
    env.reset()

    # Show randomized parameters
    param_name = list(dr_config.keys())[0]
    param_values = getattr(env._env.env_params, param_name)
    print(f"  ✓ Randomized '{param_name}': {param_values[:3]}... (showing first 3)")

    # Sample a configuration
    env.config_space.seed(42)
    action = env.config_space.sample_configuration()

    # Test single seed
    print(f"\nTesting single seed...")
    _, objectives_single, _, _, _ = env.step(action, seed=42)
    print(f"  ✓ Single seed reward: {objectives_single['reward_mean']:.2f}")

    # Test multi-seed
    print(f"\nTesting multi-seed ({n_seeds} seeds)...")
    env.reset()
    seeds = list(range(10, 10 + n_seeds))
    _, objectives_multi, _, _, _ = env.step(action, seed=seeds)

    print(f"  ✓ Multi-seed rewards:")
    for i, seed in enumerate(seeds):
        print(f"     Seed {seed}: {objectives_multi['reward_mean'][i]:.2f}")

    mean_reward = float(objectives_multi["reward_mean"].mean())
    std_reward = float(objectives_multi["reward_mean"].std())
    print(f"  ✓ Mean: {mean_reward:.2f} ± {std_reward:.2f}")

    return mean_reward


def main():
    """Run all tests."""
    print("=" * 70)
    print("Domain Randomization Test Suite for All CC Environments")
    print("=" * 70)

    # Define DR configs for each environment
    configs = {
        "CartPole-v1": {
            "masspole": {
                "distribution": "uniform",
                "min_value": 0.05,
                "max_value": 0.15,
            }
        },
        "Pendulum-v1": {
            "g": {
                "distribution": "uniform",
                "min_value": 8.0,
                "max_value": 12.0,
            }
        },
        "Acrobot-v1": {
            "link_mass_1": {
                "distribution": "normal",
                "mean": 1.0,
                "std": 0.2,
                "clip_min": 0.7,
                "clip_max": 1.3,
            }
        },
        "MountainCar-v0": {
            "force": {
                "distribution": "normal",
                "mean": 0.001,
                "std": 0.0002,
                "clip_min": 0.0007,
                "clip_max": 0.0013,
            }
        },
        "MountainCarContinuous-v0": {
            "power": {
                "distribution": "normal",
                "mean": 0.0015,
                "std": 0.0003,
                "clip_min": 0.00105,
                "clip_max": 0.00195,
            }
        },
    }

    results = {}
    for env_name, dr_config in configs.items():
        try:
            mean_reward = test_environment(env_name, dr_config)
            results[env_name] = mean_reward
            print(f"\n✅ {env_name} PASSED")
        except Exception as e:
            print(f"\n❌ {env_name} FAILED: {e}")
            results[env_name] = None

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for v in results.values() if v is not None)
    total = len(results)
    print(f"\nTests Passed: {passed}/{total}\n")

    for env_name, mean_reward in results.items():
        status = "✅" if mean_reward is not None else "❌"
        reward_str = f"{mean_reward:.2f}" if mean_reward is not None else "FAILED"
        print(f"  {status} {env_name:30s} Mean Reward: {reward_str}")

    print("\n" + "=" * 70)
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")
    print("=" * 70)


if __name__ == "__main__":
    main()
