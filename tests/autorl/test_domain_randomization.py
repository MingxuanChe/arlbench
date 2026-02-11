"""Test for domain randomization with AutoRLEnv and multi-seed pipeline."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from arlbench import AutoRLEnv


def test_autorl_env_with_domain_randomization():
    """Test that AutoRLEnv works with domain randomization."""
    jax.config.update("jax_enable_x64", True)

    config = {
        "seed": 42,
        "env_framework": "gymnax",
        "env_name": "CartPole-v1",
        "n_envs": 8,
        "algorithm": "dqn",
        "cnn_policy": False,
        "n_total_timesteps": 1e3,  # Short for testing
        "n_eval_steps": 2,
        "checkpoint": [],
        "objectives": ["reward_mean"],
        "state_features": [],
        "n_steps": 2,
        "domain_randomization": {
            "masspole": {
                "distribution": "uniform",
                "min_value": 0.05,
                "max_value": 0.15
            }
        },
        "eval_domain_randomization": {
            "masspole": {
                "distribution": "uniform",
                "min_value": 0.05,
                "max_value": 0.15
            }
        },
    }

    env = AutoRLEnv(config=config)
    env.reset()

    # Verify that the training env has domain randomization
    assert env._env._use_domain_randomization
    assert isinstance(env._env.env_params.masspole, jnp.ndarray)
    assert env._env.env_params.masspole.shape == (8,)
    assert jnp.all(env._env.env_params.masspole >= 0.05)
    assert jnp.all(env._env.env_params.masspole <= 0.15)

    # Verify that the eval env also has domain randomization
    assert env._eval_env._use_domain_randomization
    assert isinstance(env._eval_env.env_params.masspole, jnp.ndarray)

    # Sample a configuration and run a step
    env.config_space.seed(config["seed"])
    action = env.config_space.sample_configuration()

    _, objectives, _, _, _ = env.step(action, seed=42)

    # Check that we got valid objectives
    assert "reward_mean" in objectives
    assert isinstance(objectives["reward_mean"], (int, float, np.ndarray, jnp.ndarray))


def test_multi_seed_with_domain_randomization():
    """Test multi-seed pipeline with domain randomization."""
    jax.config.update("jax_enable_x64", True)

    config = {
        "seed": 42,
        "env_framework": "gymnax",
        "env_name": "CartPole-v1",
        "n_envs": 4,
        "algorithm": "dqn",
        "cnn_policy": False,
        "n_total_timesteps": 5e2,  # Very short for testing
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
    env.reset()

    # Sample a configuration
    env.config_space.seed(config["seed"])
    action = env.config_space.sample_configuration()

    # Test multi-seed
    seeds = [10, 11, 12]
    _, objectives_multi, _, _, _ = env.step(action, seed=seeds)

    # Check that we got objectives for each seed
    assert "reward_mean" in objectives_multi
    assert hasattr(objectives_multi["reward_mean"], "__len__")
    assert len(objectives_multi["reward_mean"]) == len(seeds)

    # Verify results are valid
    for reward in objectives_multi["reward_mean"]:
        assert isinstance(reward, (int, float, np.ndarray, jnp.ndarray))
        # CartPole rewards should be positive
        assert float(reward) >= 0


def test_domain_randomization_creates_different_envs_per_seed():
    """Test that each seed gets different randomized environment parameters."""
    jax.config.update("jax_enable_x64", True)

    config = {
        "seed": 42,
        "env_framework": "gymnax",
        "env_name": "CartPole-v1",
        "n_envs": 4,
        "algorithm": "dqn",
        "cnn_policy": False,
        "n_total_timesteps": 5e2,
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
            }
        },
    }

    # Create two environments with different seeds
    env1 = AutoRLEnv(config=config)
    env1.reset()

    config2 = config.copy()
    config2["seed"] = 43
    env2 = AutoRLEnv(config=config2)
    env2.reset()

    # Check that the environments have different randomized parameters
    # (due to different seeds in make_env)
    masspole1 = env1._env.env_params.masspole
    masspole2 = env2._env.env_params.masspole

    assert not jnp.allclose(masspole1, masspole2)


if __name__ == "__main__":
    test_autorl_env_with_domain_randomization()
    print("✓ test_autorl_env_with_domain_randomization passed")

    test_multi_seed_with_domain_randomization()
    print("✓ test_multi_seed_with_domain_randomization passed")

    test_domain_randomization_creates_different_envs_per_seed()
    print("✓ test_domain_randomization_creates_different_envs_per_seed passed")

    print("\nAll tests passed!")
