"""Tests for domain randomization integration with gymnax environments."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arlbench.core.environments import make_env


class TestGymnaxDomainRandomization:
    """Tests for domain randomization with gymnax environments."""

    def test_no_domain_randomization(self):
        """Test that normal environment creation still works."""
        env = make_env(
            env_framework="gymnax",
            env_name="CartPole-v1",
            n_envs=4,
            seed=42
        )

        # Check that env_params is not a pytree (single set of params)
        assert not isinstance(env.env_params.masspole, jnp.ndarray)
        assert env.env_params.masspole == 0.1
        assert env.env_params.length == 0.5

    def test_domain_randomization_uniform(self):
        """Test domain randomization with uniform distribution."""
        domain_randomization = {
            "masspole": {
                "distribution": "uniform",
                "min_value": 0.05,
                "max_value": 0.15
            }
        }

        env = make_env(
            env_framework="gymnax",
            env_name="CartPole-v1",
            n_envs=8,
            seed=42,
            domain_randomization=domain_randomization
        )

        # Check that env_params is now a pytree with different values for each env
        assert isinstance(env.env_params.masspole, jnp.ndarray)
        assert env.env_params.masspole.shape == (8,)

        # Check that all values are within the specified range
        assert jnp.all(env.env_params.masspole >= 0.05)
        assert jnp.all(env.env_params.masspole <= 0.15)

        # Check that we got different values
        assert jnp.std(env.env_params.masspole) > 0.01

        # Check that length is unchanged (single value across all envs)
        assert isinstance(env.env_params.length, jnp.ndarray)
        assert jnp.all(env.env_params.length == 0.5)

    def test_domain_randomization_normal(self):
        """Test domain randomization with normal distribution."""
        domain_randomization = {
            "length": {
                "distribution": "normal",
                "mean": 0.5,
                "std": 0.1,
                "clip_min": 0.3,
                "clip_max": 0.7
            }
        }

        env = make_env(
            env_framework="gymnax",
            env_name="CartPole-v1",
            n_envs=100,
            seed=42,
            domain_randomization=domain_randomization
        )

        # Check that env_params is now a pytree
        assert isinstance(env.env_params.length, jnp.ndarray)
        assert env.env_params.length.shape == (100,)

        # Check that all values are within clip range
        assert jnp.all(env.env_params.length >= 0.3)
        assert jnp.all(env.env_params.length <= 0.7)

        # Check that we got reasonable variation
        assert 0.40 <= jnp.mean(env.env_params.length) <= 0.60
        assert jnp.std(env.env_params.length) > 0.05

    def test_domain_randomization_multiple_params(self):
        """Test domain randomization with multiple parameters."""
        domain_randomization = {
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
        }

        env = make_env(
            env_framework="gymnax",
            env_name="CartPole-v1",
            n_envs=10,
            seed=42,
            domain_randomization=domain_randomization
        )

        # Check that both parameters are randomized
        assert isinstance(env.env_params.masspole, jnp.ndarray)
        assert isinstance(env.env_params.length, jnp.ndarray)

        assert jnp.all(env.env_params.masspole >= 0.05)
        assert jnp.all(env.env_params.masspole <= 0.15)

        assert jnp.all(env.env_params.length >= 0.3)
        assert jnp.all(env.env_params.length <= 0.7)

    def test_different_seeds_get_different_params(self):
        """Test that different seeds produce different randomized parameters."""
        domain_randomization = {
            "masspole": {
                "distribution": "uniform",
                "min_value": 0.05,
                "max_value": 0.15
            }
        }

        env1 = make_env(
            env_framework="gymnax",
            env_name="CartPole-v1",
            n_envs=8,
            seed=42,
            domain_randomization=domain_randomization
        )

        env2 = make_env(
            env_framework="gymnax",
            env_name="CartPole-v1",
            n_envs=8,
            seed=43,
            domain_randomization=domain_randomization
        )

        # Check that the two environments have different parameters
        assert not jnp.allclose(env1.env_params.masspole, env2.env_params.masspole)

    def test_env_reset_with_domain_randomization(self):
        """Test that environment reset works with domain randomization."""
        domain_randomization = {
            "masspole": {
                "distribution": "uniform",
                "min_value": 0.05,
                "max_value": 0.15
            }
        }

        env = make_env(
            env_framework="gymnax",
            env_name="CartPole-v1",
            n_envs=4,
            seed=42,
            domain_randomization=domain_randomization
        )

        rng = jax.random.key(0)
        env_state, obs = env.reset(rng)

        # Check that we got valid observations and states
        assert obs.shape == (4, 4)  # 4 envs, 4 obs dimensions
        assert env_state is not None

    def test_env_step_with_domain_randomization(self):
        """Test that environment step works with domain randomization."""
        domain_randomization = {
            "masspole": {
                "distribution": "uniform",
                "min_value": 0.05,
                "max_value": 0.15
            }
        }

        env = make_env(
            env_framework="gymnax",
            env_name="CartPole-v1",
            n_envs=4,
            seed=42,
            domain_randomization=domain_randomization
        )

        rng = jax.random.key(0)
        env_state, obs = env.reset(rng)

        # Take a step
        rng, step_rng = jax.random.split(rng)
        action_rng = jax.random.split(step_rng, 4)
        actions = jax.vmap(env.action_space.sample)(action_rng)

        env_state, (obs, reward, done, info) = env.step(env_state, actions, rng)

        # Check that we got valid outputs
        assert obs.shape == (4, 4)
        assert reward.shape == (4,)
        assert done.shape == (4,)

    def test_static_env_params_with_domain_randomization(self):
        """Test that static env_params work together with domain randomization."""
        env_params = {
            "gravity": 5.0  # Override gravity
        }

        domain_randomization = {
            "masspole": {
                "distribution": "uniform",
                "min_value": 0.05,
                "max_value": 0.15
            }
        }

        env = make_env(
            env_framework="gymnax",
            env_name="CartPole-v1",
            n_envs=4,
            seed=42,
            env_params=env_params,
            domain_randomization=domain_randomization
        )

        # Check that gravity is set to the static value
        assert jnp.all(env.env_params.gravity == 5.0)

        # Check that masspole is randomized
        assert isinstance(env.env_params.masspole, jnp.ndarray)
        assert jnp.all(env.env_params.masspole >= 0.05)
        assert jnp.all(env.env_params.masspole <= 0.15)
