"""Comprehensive tests for domain randomization across all gymnax classic control environments."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arlbench.core.environments import make_env


class TestAllGymnaxEnvironmentsDR:
    """Test domain randomization for all classic control environments."""

    @pytest.mark.parametrize(
        "env_name,param_config",
        [
            (
                "CartPole-v1",
                {
                    "masspole": {
                        "distribution": "uniform",
                        "min_value": 0.05,
                        "max_value": 0.15,
                    }
                },
            ),
            (
                "Pendulum-v1",
                {
                    "g": {
                        "distribution": "uniform",
                        "min_value": 8.0,
                        "max_value": 12.0,
                    }
                },
            ),
            (
                "Acrobot-v1",
                {
                    "link_mass_1": {
                        "distribution": "normal",
                        "mean": 1.0,
                        "std": 0.2,
                        "clip_min": 0.7,
                        "clip_max": 1.3,
                    }
                },
            ),
            (
                "MountainCar-v0",
                {
                    "force": {
                        "distribution": "normal",
                        "mean": 0.001,
                        "std": 0.0002,
                        "clip_min": 0.0007,
                        "clip_max": 0.0013,
                    }
                },
            ),
            (
                "MountainCarContinuous-v0",
                {
                    "power": {
                        "distribution": "normal",
                        "mean": 0.0015,
                        "std": 0.0003,
                        "clip_min": 0.00105,
                        "clip_max": 0.00195,
                    }
                },
            ),
        ],
    )
    def test_environment_with_dr(self, env_name, param_config):
        """Test that domain randomization works for each environment."""
        env = make_env(
            env_framework="gymnax",
            env_name=env_name,
            n_envs=8,
            seed=42,
            domain_randomization=param_config,
        )

        # Get the parameter name being randomized
        param_name = list(param_config.keys())[0]

        # Check that the parameter is randomized across environments
        param_values = getattr(env.env_params, param_name)
        assert isinstance(param_values, jnp.ndarray)
        assert param_values.shape == (8,)

        # Check that values are within expected range
        config = param_config[param_name]
        if config["distribution"] == "uniform":
            assert jnp.all(param_values >= config["min_value"])
            assert jnp.all(param_values <= config["max_value"])
        elif config["distribution"] == "normal":
            assert jnp.all(param_values >= config["clip_min"])
            assert jnp.all(param_values <= config["clip_max"])

        # Check that we got variation
        assert jnp.std(param_values) > 0.0

    @pytest.mark.parametrize(
        "env_name",
        ["CartPole-v1", "Pendulum-v1", "Acrobot-v1", "MountainCar-v0", "MountainCarContinuous-v0"],
    )
    def test_environment_reset_and_step(self, env_name):
        """Test that environments can reset and step with domain randomization."""
        # Simple DR config for each environment
        dr_configs = {
            "CartPole-v1": {"masspole": {"distribution": "uniform", "min_value": 0.05, "max_value": 0.15}},
            "Pendulum-v1": {"g": {"distribution": "uniform", "min_value": 8.0, "max_value": 12.0}},
            "Acrobot-v1": {"link_mass_1": {"distribution": "uniform", "min_value": 0.8, "max_value": 1.2}},
            "MountainCar-v0": {"force": {"distribution": "uniform", "min_value": 0.0008, "max_value": 0.0012}},
            "MountainCarContinuous-v0": {"power": {"distribution": "uniform", "min_value": 0.0012, "max_value": 0.0018}},
        }

        env = make_env(
            env_framework="gymnax",
            env_name=env_name,
            n_envs=4,
            seed=42,
            domain_randomization=dr_configs[env_name],
        )

        # Test reset
        rng = jax.random.key(0)
        env_state, obs = env.reset(rng)
        assert obs is not None
        assert env_state is not None

        # Test step
        rng, step_rng = jax.random.split(rng)
        action_rng = jax.random.split(step_rng, 4)
        actions = jax.vmap(env.action_space.sample)(action_rng)

        env_state, (obs, reward, done, info) = env.step(env_state, actions, rng)

        # Check outputs have correct shape
        assert obs.shape[0] == 4
        assert reward.shape == (4,)
        assert done.shape == (4,)


class TestMultiParameterDR:
    """Test environments with multiple parameters randomized."""

    def test_cartpole_multi_param(self):
        """Test CartPole with multiple parameters randomized."""
        domain_randomization = {
            "masspole": {
                "distribution": "uniform",
                "min_value": 0.05,
                "max_value": 0.15,
            },
            "length": {
                "distribution": "normal",
                "mean": 0.5,
                "std": 0.1,
                "clip_min": 0.3,
                "clip_max": 0.7,
            },
            "gravity": {
                "distribution": "uniform",
                "min_value": 8.0,
                "max_value": 11.0,
            },
        }

        env = make_env(
            env_framework="gymnax",
            env_name="CartPole-v1",
            n_envs=10,
            seed=42,
            domain_randomization=domain_randomization,
        )

        # Check all three parameters are randomized
        assert isinstance(env.env_params.masspole, jnp.ndarray)
        assert isinstance(env.env_params.length, jnp.ndarray)
        assert isinstance(env.env_params.gravity, jnp.ndarray)

        # Check ranges
        assert jnp.all(env.env_params.masspole >= 0.05)
        assert jnp.all(env.env_params.masspole <= 0.15)
        assert jnp.all(env.env_params.length >= 0.3)
        assert jnp.all(env.env_params.length <= 0.7)
        assert jnp.all(env.env_params.gravity >= 8.0)
        assert jnp.all(env.env_params.gravity <= 11.0)

    def test_pendulum_multi_param(self):
        """Test Pendulum with multiple parameters randomized."""
        domain_randomization = {
            "max_torque": {
                "distribution": "uniform",
                "min_value": 1.6,
                "max_value": 2.4,
            },
            "g": {
                "distribution": "uniform",
                "min_value": 8.0,
                "max_value": 12.0,
            },
            "m": {
                "distribution": "normal",
                "mean": 1.0,
                "std": 0.2,
                "clip_min": 0.7,
                "clip_max": 1.3,
            },
        }

        env = make_env(
            env_framework="gymnax",
            env_name="Pendulum-v1",
            n_envs=10,
            seed=42,
            domain_randomization=domain_randomization,
        )

        # Check all parameters are randomized
        assert isinstance(env.env_params.max_torque, jnp.ndarray)
        assert isinstance(env.env_params.g, jnp.ndarray)
        assert isinstance(env.env_params.m, jnp.ndarray)

        # Check ranges
        assert jnp.all(env.env_params.max_torque >= 1.6)
        assert jnp.all(env.env_params.max_torque <= 2.4)
        assert jnp.all(env.env_params.g >= 8.0)
        assert jnp.all(env.env_params.g <= 12.0)
        assert jnp.all(env.env_params.m >= 0.7)
        assert jnp.all(env.env_params.m <= 1.3)

    def test_acrobot_multi_param(self):
        """Test Acrobot with multiple parameters randomized."""
        domain_randomization = {
            "link_length_1": {
                "distribution": "uniform",
                "min_value": 0.8,
                "max_value": 1.2,
            },
            "link_length_2": {
                "distribution": "uniform",
                "min_value": 0.8,
                "max_value": 1.2,
            },
            "link_mass_1": {
                "distribution": "normal",
                "mean": 1.0,
                "std": 0.2,
                "clip_min": 0.7,
                "clip_max": 1.3,
            },
        }

        env = make_env(
            env_framework="gymnax",
            env_name="Acrobot-v1",
            n_envs=10,
            seed=42,
            domain_randomization=domain_randomization,
        )

        # Check all parameters are randomized
        assert isinstance(env.env_params.link_length_1, jnp.ndarray)
        assert isinstance(env.env_params.link_length_2, jnp.ndarray)
        assert isinstance(env.env_params.link_mass_1, jnp.ndarray)

        # Check ranges
        assert jnp.all(env.env_params.link_length_1 >= 0.8)
        assert jnp.all(env.env_params.link_length_1 <= 1.2)
        assert jnp.all(env.env_params.link_length_2 >= 0.8)
        assert jnp.all(env.env_params.link_length_2 <= 1.2)
        assert jnp.all(env.env_params.link_mass_1 >= 0.7)
        assert jnp.all(env.env_params.link_mass_1 <= 1.3)
