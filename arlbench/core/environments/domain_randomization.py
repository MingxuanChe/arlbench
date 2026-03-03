"""Domain randomization for environment parameters."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any
from dataclasses import replace

import jax
import jax.numpy as jnp
from chex import PRNGKey

if TYPE_CHECKING:
    pass


class DomainRandomizationConfig:
    """Configuration for domain randomization of environment parameters."""

    def __init__(
        self,
        param_name: str,
        distribution: str = "uniform",
        min_value: float | None = None,
        max_value: float | None = None,
        mean: float | None = None,
        std: float | None = None,
        clip_min: float | None = None,
        clip_max: float | None = None,
    ):
        """Initialize domain randomization configuration for a single parameter.

        Args:
            param_name: Name of the environment parameter to randomize
            distribution: Type of distribution ("uniform" or "normal")
            min_value: Minimum value for uniform distribution
            max_value: Maximum value for uniform distribution
            mean: Mean value for normal distribution
            std: Standard deviation for normal distribution
            clip_min: Minimum clipping value (for normal distribution)
            clip_max: Maximum clipping value (for normal distribution)
        """
        self.param_name = param_name
        self.distribution = distribution

        if distribution == "uniform":
            if min_value is None or max_value is None:
                raise ValueError(
                    "min_value and max_value must be specified for uniform distribution"
                )
            self.min_value = min_value
            self.max_value = max_value
        elif distribution == "normal":
            if mean is None or std is None:
                raise ValueError("mean and std must be specified for normal distribution")
            self.mean = mean
            self.std = std
            self.clip_min = clip_min
            self.clip_max = clip_max
        else:
            raise ValueError(
                f"Unknown distribution: {distribution}. Must be 'uniform' or 'normal'"
            )

    def sample(self, rng: PRNGKey) -> float:
        """Sample a value from the configured distribution.

        Args:
            rng: JAX random key

        Returns:
            Sampled value
        """
        if self.distribution == "uniform":
            return jax.random.uniform(
                rng, minval=self.min_value, maxval=self.max_value
            )
        elif self.distribution == "normal":
            sample = self.mean + self.std * jax.random.normal(rng)
            if self.clip_min is not None or self.clip_max is not None:
                return jnp.clip(sample, self.clip_min, self.clip_max)
            return sample
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> DomainRandomizationConfig:
        """Create a DomainRandomizationConfig from a dictionary.

        Args:
            config_dict: Dictionary with configuration parameters

        Returns:
            DomainRandomizationConfig instance
        """
        return cls(**config_dict)


class DomainRandomizer:
    """Domain randomizer for sampling environment parameters from distributions."""

    def __init__(self, randomization_configs: list[DomainRandomizationConfig]):
        """Initialize domain randomizer.

        Args:
            randomization_configs: List of randomization configurations for each parameter
        """
        self.randomization_configs = randomization_configs

    def sample_env_params(self, base_env_params: Any, rng: PRNGKey) -> Any:
        """Sample environment parameters from configured distributions.

        Args:
            base_env_params: Base environment parameters (e.g., from gymnax.make)
            rng: JAX random key

        Returns:
            Environment parameters with randomized values
        """
        if len(self.randomization_configs) == 0:
            return base_env_params

        # Split RNG for each parameter
        rngs = jax.random.split(rng, len(self.randomization_configs))

        # Sample and update parameters
        updates = {}
        for config, param_rng in zip(self.randomization_configs, rngs):
            updates[config.param_name] = config.sample(param_rng)

        return replace(base_env_params, **updates)

    @classmethod
    def from_config_dict(cls, config: dict[str, Any]) -> DomainRandomizer | None:
        """Create a DomainRandomizer from a configuration dictionary.

        Args:
            config: Dictionary with domain randomization configuration.
                   Expected format:
                   {
                       "param_name1": {
                           "distribution": "uniform",
                           "min_value": 0.05,
                           "max_value": 0.15
                       },
                       "param_name2": {
                           "distribution": "normal",
                           "mean": 1.0,
                           "std": 0.2,
                           "clip_min": 0.5,
                           "clip_max": 1.5
                       }
                   }

        Returns:
            DomainRandomizer instance or None if config is empty
        """
        if not config:
            return None

        randomization_configs = []
        for param_name, param_config in config.items():
            # Convert OmegaConf DictConfig to plain dict before mutating
            param_config = dict(param_config)
            param_config["param_name"] = param_name
            randomization_configs.append(
                DomainRandomizationConfig.from_dict(param_config)
            )

        return cls(randomization_configs)
