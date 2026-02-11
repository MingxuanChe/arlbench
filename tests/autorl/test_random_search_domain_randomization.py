"""Test random search with domain randomization."""
from __future__ import annotations

import jax

from arlbench import AutoRLEnv


def test_random_search_with_domain_randomization():
    """Test that random search works with domain randomization."""
    jax.config.update("jax_enable_x64", True)

    config = {
        "seed": 42,  # Single seed for init, we'll use list in step()
        "env_framework": "gymnax",
        "env_name": "CartPole-v1",
        "n_envs": 4,
        "algorithm": "dqn",
        "cnn_policy": False,
        "n_total_timesteps": 1e3,  # Short for testing
        "n_eval_steps": 2,
        "checkpoint": [],
        "objectives": ["reward_mean"],
        "state_features": [],
        "n_steps": 1,
        "domain_randomization": {
            "masspole": {
                "distribution": "uniform",
                "min_value": 0.05,
                "max_value": 0.15
            },
            "length": {
                "distribution": "normal",
                "mean": 0.5,
                "std": 0.1,
                "clip_min": 0.3,
                "clip_max": 0.7
            }
        },
        "eval_domain_randomization": {
            "masspole": {
                "distribution": "uniform",
                "min_value": 0.05,
                "max_value": 0.15
            },
            "length": {
                "distribution": "normal",
                "mean": 0.5,
                "std": 0.1,
                "clip_min": 0.3,
                "clip_max": 0.7
            }
        },
    }

    env = AutoRLEnv(config=config)

    # Simulate random search: multiple configurations
    n_configs = 3
    env.config_space.seed(42)

    # Seeds for multi-seed evaluation
    eval_seeds = [1, 2, 3]

    results = []
    for i in range(n_configs):
        env.reset()
        action = env.config_space.sample_configuration()

        # Use multi-seed evaluation
        _, objectives, _, _, _ = env.step(action, seed=eval_seeds)

        # Objectives should be arrays (one value per seed)
        assert hasattr(objectives["reward_mean"], "__len__")
        assert len(objectives["reward_mean"]) == len(eval_seeds)

        # Store mean reward across seeds
        mean_reward = float(objectives["reward_mean"].mean())
        results.append(mean_reward)

        print(f"Config {i+1}: Mean reward = {mean_reward:.2f}")

    # Check that we got valid results
    assert len(results) == n_configs
    assert all(r >= 0 for r in results)

    print(f"\nBest config achieved mean reward: {max(results):.2f}")
    return results


if __name__ == "__main__":
    results = test_random_search_with_domain_randomization()
    print(f"\n✓ Random search with domain randomization completed successfully")
    print(f"  Results across {len(results)} configs: {[f'{r:.2f}' for r in results]}")
