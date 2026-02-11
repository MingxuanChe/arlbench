"""
Example: Domain Randomization with ARLBench

This example demonstrates how to use domain randomization for robust RL training.
We'll train a DQN agent on CartPole with randomized pole mass and length.
"""

import jax

jax.config.update("jax_enable_x64", True)

from arlbench import AutoRLEnv


def main():
    print("=" * 80)
    print("Domain Randomization Example for ARLBench")
    print("=" * 80)

    # Configuration with domain randomization
    config = {
        "seed": 42,
        "env_framework": "gymnax",
        "env_name": "CartPole-v1",
        "n_envs": 8,  # 8 parallel environments
        "algorithm": "dqn",
        "cnn_policy": False,
        "n_total_timesteps": 5e4,
        "n_eval_steps": 10,
        "n_eval_episodes": 10,
        "checkpoint": [],
        "objectives": ["reward_mean"],
        "state_features": [],
        "n_steps": 1,
        # Domain randomization for training
        "domain_randomization": {
            "masspole": {
                "distribution": "uniform",
                "min_value": 0.05,  # 50% of default (0.1)
                "max_value": 0.15,  # 150% of default
            },
            "length": {
                "distribution": "normal",
                "mean": 0.5,  # default value
                "std": 0.1,  # 20% std
                "clip_min": 0.3,  # minimum safe value
                "clip_max": 0.7,  # maximum safe value
            },
        },
        # Same randomization for evaluation
        "eval_domain_randomization": {
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
        },
    }

    print("\n1. Creating AutoRLEnv with domain randomization...")
    env = AutoRLEnv(config=config)
    env.reset()

    print(f"Training environment created with {config['n_envs']} parallel envs")
    print(f"Each env has randomized parameters:")
    print(f"     - masspole: {env._env.env_params.masspole[:3]}... (showing first 3)")
    print(f"     - length: {env._env.env_params.length[:3]}...")

    print("\n2. Sampling a hyperparameter configuration...")
    env.config_space.seed(42)
    hp_config = env.config_space.sample_configuration()
    print(f"Sampled config with learning_rate={hp_config['learning_rate']:.6f}")

    print("\n3. Training with a single seed...")
    _, objectives_single, _, _, _ = env.step(hp_config, seed=42)
    print(f"Single seed (42) reward: {objectives_single['reward_mean']:.2f}")

    print("\n4. Training with multiple seeds...")
    env.reset()
    eval_seeds = [10, 11, 12, 13, 14]
    _, objectives_multi, _, _, _ = env.step(hp_config, seed=eval_seeds)

    print(f"Multi-seed rewards:")
    for i, seed in enumerate(eval_seeds):
        print(f"     Seed {seed}: {objectives_multi['reward_mean'][i]:.2f}")

    mean_reward = objectives_multi["reward_mean"].mean()
    std_reward = objectives_multi["reward_mean"].std()
    print(f"Mean: {mean_reward:.2f} ± {std_reward:.2f}")

    print("\n5. Simulating random search across 5 configurations...")
    best_config = None
    best_reward = float("-inf")

    env.config_space.seed(100)
    for i in range(5):
        env.reset()
        hp_config = env.config_space.sample_configuration()

        # Evaluate with multiple seeds for robustness
        _, objectives, _, _, _ = env.step(hp_config, seed=[1, 2, 3])
        mean_reward = float(objectives["reward_mean"].mean())

        print(f"   Config {i+1}: reward = {mean_reward:.2f}")

        if mean_reward > best_reward:
            best_reward = mean_reward
            best_config = hp_config

    print(f"\nBest configuration found:")
    print(f"     Learning rate: {best_config['learning_rate']:.6f}")
    print(f"     Mean reward: {best_reward:.2f}")

    print("\n" + "=" * 80)
    print("Domain randomization example completed successfully!")
    print("=" * 80)

    print("\nKey Benefits:")
    print("  • Each parallel env has different randomized parameters")
    print("  • Each seed gets a fresh set of randomized envs")
    print("  • Evaluates generalization within the parameter distribution")
    print("  • More robust to environment variations")


if __name__ == "__main__":
    main()
