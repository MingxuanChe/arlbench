"""Gymnax environment adapter."""
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import gymnax
import jax
import jax.numpy as jnp
from dataclasses import replace

from .autorl_env import Environment
from .domain_randomization import DomainRandomizer

if TYPE_CHECKING:
    from chex import PRNGKey


class GymnaxEnv(Environment):
    """A gymnax-based RL environment."""

    def __init__(
        self,
        env_name: str,
        n_envs: int,
        env_kwargs: dict[str, Any] | None = None,
        env_params: dict[str, Any] | None = None,
        domain_randomization: dict[str, Any] | None = None,
        seed: int = 0,
    ):
        """Creates a gymnax environment for JAX-based RL training.

        Args:
            env_name (str): Name/id of the brax environment.
            n_envs (int): Number of environments.
            env_kwargs (dict[str, Any] | None, optional): Keyword arguments
                to pass to the gymnax environment. Defaults to None.
            env_params (dict[str, Any] | None, optional): Static environment parameters
                to override defaults. Defaults to None.
            domain_randomization (dict[str, Any] | None, optional): Domain randomization
                configuration. Defaults to None.
            seed (int, optional): Random seed for domain randomization. Defaults to 0.
        """
        if env_kwargs is None:
            env_kwargs = {}
        env, og_env_params = gymnax.make(env_name, **env_kwargs)
        base_env_params = replace(og_env_params, **(env_params or {}))
        super().__init__(env_name, env, n_envs)

        # Domain randomization setup
        self.domain_randomizer = DomainRandomizer.from_config_dict(
            domain_randomization or {}
        )
        self._use_domain_randomization = self.domain_randomizer is not None

        if self._use_domain_randomization:
            # Sample different env_params for each environment
            rng = jax.random.key(seed)
            param_rngs = jax.random.split(rng, n_envs)
            # Vectorize the sampling for all environments
            self.env_params = jax.vmap(
                lambda r: self.domain_randomizer.sample_env_params(base_env_params, r)
            )(param_rngs)
        else:
            self.env_params = base_env_params

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, rng: PRNGKey):
        """Resets the environment."""
        reset_rng = jax.random.split(rng, self.n_envs)
        if self._use_domain_randomization:
            # Each environment has its own params
            obs, env_state = jax.vmap(self._env.reset, in_axes=(0, 0))(
                reset_rng, self.env_params
            )
        else:
            # All environments share the same params
            obs, env_state = jax.vmap(self._env.reset, in_axes=(0, None))(
                reset_rng, self.env_params
            )
        return env_state, obs

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, env_state: Any, action: Any, rng: PRNGKey):
        """Steps the environment forward."""
        step_rng = jax.random.split(rng, self.n_envs)
        if self._use_domain_randomization:
            # Each environment has its own params
            obs, env_state, reward, done, info = jax.vmap(
                self._env.step, in_axes=(0, 0, 0, 0)
            )(step_rng, env_state, action, self.env_params)
        else:
            # All environments share the same params
            obs, env_state, reward, done, info = jax.vmap(
                self._env.step, in_axes=(0, 0, 0, None)
            )(step_rng, env_state, action, self.env_params)

        return env_state, (obs, reward, done, info)

    @property
    def action_space(self):
        """Action space of the environment."""
        if self._use_domain_randomization:
            # Use the first environment's params for action space; only for initialization
            return self._env.action_space(jax.tree_util.tree_map(lambda x: x[0], self.env_params))
        return self._env.action_space(self.env_params)

    @functools.partial(jax.jit, static_argnums=0)
    def sample_action(self, rng: PRNGKey):
        """Samples a random action from the action space."""
        return self.action_space.sample(rng)

    @property
    def observation_space(self):
        """Observation space of the environment."""
        if self._use_domain_randomization:
            # Use the first environment's params for observation space; only for initialization
            return self._env.observation_space(
                jax.tree_util.tree_map(lambda x: x[0], self.env_params)
            )
        return self._env.observation_space(self.env_params)

