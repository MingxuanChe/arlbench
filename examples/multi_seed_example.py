import jax
jax.config.update("jax_enable_x64", True)

from arlbench import AutoRLEnv
import time

def test_multi_seed():
    config = {
        "seed": 42,
        "env_framework": "gymnax",
        "env_name": "CartPole-v1",
        "n_envs": 10,
        "algorithm": "dqn",
        "cnn_policy": False,
        "n_total_timesteps": 1e4,  # Reduced for speed
        "n_eval_steps": 5,
        "checkpoint": [],
        "objectives": ["reward_mean", "runtime"],
        "state_features": [],
        "n_steps": 2,
    }

    print("Initializing AutoRLEnv...")
    env = AutoRLEnv(config=config)
    env.reset()
    
    # Seed the configuration space for reproducibility
    env.config_space.seed(config["seed"])
    action = env.config_space.sample_configuration()
    
    # Test single seed
    print("\nTesting single seed (42)...")
    start_time = time.time()
    _, objectives, _, _, _= env.step(action, seed=42)
    print(f"Single seed time: {time.time() - start_time:.2f}s")
    print(f"Objectives: {objectives}")
    
    # Test multi seed
    seeds = [42, 43, 44]
    print(f"\nTesting multi seed {seeds}...")
    start_time = time.time()
    obs_multi, objectives_multi, _, trunc_multi, info_multi = env.step(action, seed=seeds)
    print(f"Multi seed time: {time.time() - start_time:.2f}s")
    print(f"Objectives (per seed): {objectives_multi}")
    
    # Basic checks
    import numpy as np
    assert "reward_mean" in objectives_multi
    assert "runtime" in objectives_multi
    # Can be either np.ndarray or jax array
    assert hasattr(objectives_multi["reward_mean"], "__len__")
    assert len(objectives_multi["reward_mean"]) == len(seeds)
    print(f"Mean aggregated reward: {np.array(objectives_multi['reward_mean']).mean()}")
    
    # Comparison Test
    print("\nComparing Sequential vs Multi-seed execution...")
    compare_seeds = [10, 11, 12]
    
    # Sequential
    seq_rewards = []
    for s in compare_seeds:
        env.reset() # Reset to ensure fresh start for each seed
        obs, objs, _, _, _ = env.step(action, seed=s)
        seq_rewards.append(objs["reward_mean"])
    
    mean_seq_reward = sum(seq_rewards) / len(seq_rewards)
    print(f"Sequential Mean Reward: {mean_seq_reward}")

    # Multi-seed
    env.reset() # Reset to ensure fresh start
    obs, objs_multi, _, _, info_multi = env.step(action, seed=compare_seeds)
    multi_rewards = objs_multi["reward_mean"]  # Now an array
    print(f"Multi-seed Rewards (per seed): {multi_rewards}")
    
    # Check that individual rewards match
    print("\nAsserting closeness of results...")
    print(f"Sequential Rewards: {seq_rewards}")
    print(f"Multi-seed Rewards: {multi_rewards.tolist()}")
    for i, (seq_r, multi_r) in enumerate(zip(seq_rewards, multi_rewards)):
        assert np.isclose(seq_r, multi_r, atol=1e-4), \
            f"Mismatch at seed {compare_seeds[i]}: Sequential {seq_r} != Multi {multi_r}"
    
    # Also check means
    mean_multi_reward = multi_rewards.mean()
    print(f"Mean Sequential Reward: {mean_seq_reward}, Mean Multi-seed Reward: {mean_multi_reward}")
    assert np.isclose(mean_seq_reward, mean_multi_reward, atol=1e-4), \
        f"Mismatch in means: Sequential {mean_seq_reward} != Multi {mean_multi_reward}"
    print("Comparison passed!")

    # Performance Comparison (10 seeds)
    print("\nPerformance Comparison (10 seeds)...")
    perf_seeds = list(range(100, 110))
    
    # Sequential Time
    start_time = time.time()
    for s in perf_seeds:
        env.reset()
        env.step(action, seed=s)
    seq_time = time.time() - start_time
    print(f"Sequential time for 10 seeds: {seq_time:.2f}s")
    
    # Multi-seed Time
    start_time = time.time()
    env.reset()
    env.step(action, seed=perf_seeds)
    multi_time = time.time() - start_time
    print(f"Multi-seed time for 10 seeds: {multi_time:.2f}s")
    
    print(f"Speedup: {seq_time / multi_time:.2f}x")

    print("\nTest passed!")

if __name__ == "__main__":
    test_multi_seed()
