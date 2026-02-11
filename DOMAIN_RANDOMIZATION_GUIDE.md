# Domain Randomization in ARLBench

This guide explains how to use domain randomization for training robust RL agents in ARLBench.

## Overview

Domain randomization is a technique for training robust RL agents by randomizing environment parameters during training. This helps the agent generalize better to variations in the environment.

In ARLBench, domain randomization is supported for gymnax environments (like CartPole-v1). Each parallel environment in the vectorized setup can have different randomized parameters sampled from specified distributions.

## Features

- **Distribution types**: 
  - Uniform distribution: `min_value`, `max_value`
  - Normal (Gaussian) distribution: `mean`, `std`, with optional `clip_min`, `clip_max`
- **Multi-parameter randomization**: Randomize multiple environment parameters simultaneously
- **Multi-seed support**: Each seed gets its own set of randomized environments
- **Separate train/eval randomization**: Configure different distributions for training and evaluation

## Configuration

### Basic Example

Here's a minimal example for CartPole-v1 with domain randomization:

```yaml
# environment config (e.g., cc_cartpole_dr.yaml)
name: "CartPole-v1"
framework: "gymnax"
n_envs: 8

domain_randomization:
  masspole:
    distribution: "uniform"
    min_value: 0.05
    max_value: 0.15
```

### Complete Example

A more comprehensive config with multiple parameters:

```yaml
name: "CartPole-v1"
framework: "gymnax"
n_total_timesteps: 1e5
n_envs: 8

# Static parameters (optional)
env_params:
  gravity: 9.8

# Training domain randomization
domain_randomization:
  masspole:
    distribution: "uniform"
    min_value: 0.05
    max_value: 0.15
  length:
    distribution: "normal"
    mean: 0.5
    std: 0.1
    clip_min: 0.3
    clip_max: 0.7

# Evaluation domain randomization
eval_domain_randomization:
  masspole:
    distribution: "uniform"
    min_value: 0.05
    max_value: 0.15
  length:
    distribution: "normal"
    mean: 0.5
    std: 0.1
    clip_min: 0.3
    clip_max: 0.7
```

## Available Environment Parameters

For CartPole-v1 (gymnax), the following parameters can be randomized:

- `gravity` (default: 9.8)
- `masscart` (default: 1.0)
- `masspole` (default: 0.1)
- `length` (default: 0.5) - half-length of the pole
- `force_mag` (default: 10.0)
- `tau` (default: 0.02) - time step
- `theta_threshold_radians` (default: ~0.209)
- `x_threshold` (default: 2.4)

To find parameters for other gymnax environments:

```python
import gymnax
env, env_params = gymnax.make("CartPole-v1")
print(env_params)
```

## Usage Examples

### Python API

```python
from arlbench import AutoRLEnv

config = {
    "seed": 42,
    "env_framework": "gymnax",
    "env_name": "CartPole-v1",
    "n_envs": 8,
    "algorithm": "dqn",
    "n_total_timesteps": 1e5,
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
    }
}

env = AutoRLEnv(config=config)
env.reset()

# Sample configuration and train
env.config_space.seed(42)
action = env.config_space.sample_configuration()

# Single seed
_, objectives, _, _, _ = env.step(action, seed=42)

# Multi-seed (each seed gets different randomized environments)
_, objectives, _, _, _ = env.step(action, seed=[42, 43, 44])
```

### Hydra Config (for running experiments with HyperSweeper)

Use HyperSweeper with domain randomization to run hyperparameter optimization experiments.

Example config file `examples/configs/rs_ppo_cartpole_dr.yaml`:

```yaml
defaults:
  - _self_
  - /algorithm: ppo
  - /environment: cc_cartpole_dr
  - search_space: ppo_cc
  - override hydra/sweeper: HyperRS

hydra:
  sweeper:
    n_trials: 20
    search_space: ${search_space}
    sweeper_kwargs:
      maximize: true
      max_parallelization: 1

autorl:
  seed: [42, 43, 44, 45, 46]  # Multi-seed evaluation
  n_total_timesteps: 50000
  objectives: ["reward_mean"]
```

#### Basic Usage

Run with default settings:

```bash
pixi run python run_arlbench.py --config-name=rs_ppo_cartpole_dr -m
```

Or using DQN:

```bash
pixi run python run_arlbench.py --config-name=rs_dqn_cartpole_dr -m
```

**Note**: The `-m` flag (or `--multirun`) is required to activate the HyperSweeper.

#### Customizing Parameters

Override specific parameters from command line:

```bash
# Adjust number of trials
pixi run python run_arlbench.py --config-name=rs_ppo_cartpole_dr -m \
  hydra.sweeper.n_trials=10

# Change seeds
pixi run python run_arlbench.py --config-name=rs_ppo_cartpole_dr -m \
  autorl.seed=42

# Use different environment (requires matching DR config)
pixi run python run_arlbench.py --config-name=rs_ppo_cartpole_dr -m \
  environment=cc_cartpole_dr

# Switch algorithm and search space
pixi run python run_arlbench.py --config-name=rs_ppo_cartpole_dr -m \
  algorithm=ppo search_space=ppo_cc

# Combine multiple overrides
pixi run python run_arlbench.py --config-name=rs_ppo_cartpole_dr -m \
  hydra.sweeper.n_trials=5 autorl.seed=42 algorithm=ppo \
  search_space=ppo_cc environment=cc_cartpole_dr
```

#### Creating Your Own DR Config

1. Create or use an environment config with domain randomization (e.g., `configs/environment/cc_cartpole_dr.yaml`)
2. Create a sweeper config that references it:

```yaml
defaults:
  - _self_
  - /algorithm: <your_algorithm>
  - /environment: <your_env_dr>
  - search_space: <your_search_space>
  - override hydra/sweeper: HyperRS

hydra:
  sweeper:
    n_trials: 20
    search_space: ${search_space}
    sweeper_kwargs:
      maximize: true
      max_parallelization: 1

autorl:
  seed: [42, 43, 44]  # List for multi-seed
  # ... other settings
```

3. Run with: `pixi run python run_arlbench.py --config-name=<your_config> -m`

## Multi-Seed Behavior

When using multi-seed training with domain randomization:

1. Each seed (e.g., 42, 43, 44) gets its own set of randomized environments
2. Within each seed, there are `n_envs` parallel environments, each with different randomized parameters
3. The randomization is deterministic based on the seed, ensuring reproducibility

Example:
- Seed 42: 8 environments with params sampled using RNG(42)
- Seed 43: 8 environments with params sampled using RNG(43)
- etc.

## Best Practices

1. **Clipping for normal distributions**: Always specify `clip_min` and `clip_max` for normal distributions to avoid extreme values that could break the environment.

2. **Reasonable ranges**: Choose ranges that keep the environment physically plausible. For CartPole:
   - `masspole`: 0.05-0.15 (default 0.1)
   - `length`: 0.3-0.7 (default 0.5)
   - `gravity`: 8.0-11.0 (default 9.8)

3. **Same distribution for eval**: Use the same distribution for training and evaluation to test generalization within the training distribution.

4. **Start simple**: Begin with randomizing one parameter, then gradually add more.

5. **Check environment remains valid**: Some parameter combinations might make the environment too easy/hard or unstable.

## Testing

Run the domain randomization tests:

```bash
# Unit tests
pixi run pytest tests/core/environments/test_domain_randomization.py -v

# Integration tests
pixi run pytest tests/core/environments/test_gymnax_domain_randomization.py -v
pixi run pytest tests/autorl/test_domain_randomization.py -v

# Random search test
pixi run python tests/autorl/test_random_search_domain_randomization.py

# Test HyperSweeper configs (quick smoke test)
./test_dr_configs.sh
```

## Limitations

- Currently only supported for gymnax environments
- All `n_envs` environments get different parameters (cannot have some randomized and some fixed)
- Randomization happens at environment creation time, not per episode

## Future Extensions

Potential future improvements:
- Support for other environment frameworks (Brax, Gymnasium)
- Per-episode randomization (resample parameters on each reset)
- Curriculum learning (gradually increase randomization range)
- Domain randomization schedules (change distribution over training)
