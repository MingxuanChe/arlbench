"""Tests for domain randomization functionality."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arlbench.core.environments.domain_randomization import (
    DomainRandomizationConfig,
    DomainRandomizer,
)


class TestDomainRandomizationConfig:
    """Tests for DomainRandomizationConfig class."""

    def test_uniform_distribution(self):
        """Test uniform distribution sampling."""
        config = DomainRandomizationConfig(
            param_name="test_param",
            distribution="uniform",
            min_value=0.5,
            max_value=1.5
        )

        # Sample multiple values
        rng = jax.random.key(42)
        samples = []
        for i in range(100):
            rng, subrng = jax.random.split(rng)
            sample = config.sample(subrng)
            samples.append(float(sample))

        # Check that all samples are within bounds
        assert all(0.5 <= s <= 1.5 for s in samples)
        # Check that we got some variation
        assert np.std(samples) > 0.1

    def test_normal_distribution(self):
        """Test normal distribution sampling."""
        config = DomainRandomizationConfig(
            param_name="test_param",
            distribution="normal",
            mean=1.0,
            std=0.2
        )

        # Sample multiple values
        rng = jax.random.key(42)
        samples = []
        for i in range(1000):
            rng, subrng = jax.random.split(rng)
            sample = config.sample(subrng)
            samples.append(float(sample))

        # Check mean and std are roughly correct
        assert abs(np.mean(samples) - 1.0) < 0.05
        assert abs(np.std(samples) - 0.2) < 0.05

    def test_normal_distribution_with_clipping(self):
        """Test normal distribution with clipping."""
        config = DomainRandomizationConfig(
            param_name="test_param",
            distribution="normal",
            mean=1.0,
            std=0.5,
            clip_min=0.5,
            clip_max=1.5
        )

        # Sample multiple values
        rng = jax.random.key(42)
        samples = []
        for i in range(1000):
            rng, subrng = jax.random.split(rng)
            sample = config.sample(subrng)
            samples.append(float(sample))

        # Check that all samples are within clip bounds
        assert all(0.5 <= s <= 1.5 for s in samples)
        # Check that we got clipping (some values at the boundaries)
        assert any(abs(s - 0.5) < 0.01 for s in samples)
        assert any(abs(s - 1.5) < 0.01 for s in samples)

    def test_invalid_distribution(self):
        """Test that invalid distribution raises error."""
        with pytest.raises(ValueError, match="Unknown distribution"):
            DomainRandomizationConfig(
                param_name="test_param",
                distribution="invalid"
            )

    def test_uniform_missing_params(self):
        """Test that uniform distribution without min/max raises error."""
        with pytest.raises(ValueError, match="min_value and max_value must be specified"):
            DomainRandomizationConfig(
                param_name="test_param",
                distribution="uniform"
            )

    def test_normal_missing_params(self):
        """Test that normal distribution without mean/std raises error."""
        with pytest.raises(ValueError, match="mean and std must be specified"):
            DomainRandomizationConfig(
                param_name="test_param",
                distribution="normal"
            )

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "param_name": "masspole",
            "distribution": "uniform",
            "min_value": 0.05,
            "max_value": 0.15
        }
        config = DomainRandomizationConfig.from_dict(config_dict)

        assert config.param_name == "masspole"
        assert config.distribution == "uniform"
        assert config.min_value == 0.05
        assert config.max_value == 0.15


class TestDomainRandomizer:
    """Tests for DomainRandomizer class."""

    def test_empty_randomizer(self):
        """Test randomizer with no configs."""
        randomizer = DomainRandomizer([])

        # Create a dummy env_params object
        from dataclasses import dataclass

        @dataclass
        class EnvParams:
            masspole: float = 0.1
            length: float = 0.5

        base_params = EnvParams()
        rng = jax.random.key(42)

        result = randomizer.sample_env_params(base_params, rng)

        # Should return unchanged params
        assert result.masspole == 0.1
        assert result.length == 0.5

    def test_single_param_randomization(self):
        """Test randomizing a single parameter."""
        config = DomainRandomizationConfig(
            param_name="masspole",
            distribution="uniform",
            min_value=0.05,
            max_value=0.15
        )
        randomizer = DomainRandomizer([config])

        from dataclasses import dataclass

        @dataclass
        class EnvParams:
            masspole: float = 0.1
            length: float = 0.5

        base_params = EnvParams()
        rng = jax.random.key(42)

        result = randomizer.sample_env_params(base_params, rng)

        # masspole should be randomized
        assert 0.05 <= result.masspole <= 0.15
        # length should remain the same
        assert result.length == 0.5

    def test_multiple_param_randomization(self):
        """Test randomizing multiple parameters."""
        configs = [
            DomainRandomizationConfig(
                param_name="masspole",
                distribution="uniform",
                min_value=0.05,
                max_value=0.15
            ),
            DomainRandomizationConfig(
                param_name="length",
                distribution="normal",
                mean=0.5,
                std=0.1,
                clip_min=0.3,
                clip_max=0.7
            )
        ]
        randomizer = DomainRandomizer(configs)

        from dataclasses import dataclass

        @dataclass
        class EnvParams:
            masspole: float = 0.1
            length: float = 0.5
            gravity: float = 9.8

        base_params = EnvParams()
        rng = jax.random.key(42)

        result = randomizer.sample_env_params(base_params, rng)

        # Both masspole and length should be randomized
        assert 0.05 <= result.masspole <= 0.15
        assert 0.3 <= result.length <= 0.7
        # gravity should remain the same
        assert result.gravity == 9.8

    def test_from_config_dict_empty(self):
        """Test creating randomizer from empty dict."""
        randomizer = DomainRandomizer.from_config_dict({})

        assert randomizer is None

    def test_from_config_dict(self):
        """Test creating randomizer from config dict."""
        config_dict = {
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
        randomizer = DomainRandomizer.from_config_dict(config_dict)

        assert randomizer is not None
        assert len(randomizer.randomization_configs) == 2

        # Check that both parameters can be found
        param_names = [c.param_name for c in randomizer.randomization_configs]
        assert "masspole" in param_names
        assert "length" in param_names
