# ARLBench Kisski Cluster Scripts

This directory contains SLURM job scripts for running ARLBench experiments on the Kisski cluster using partitioned Sobol sequences.

## Overview

- **Total Scripts**: 22 algorithm-environment combinations
- **Seeds**: 50 (42-91)
- **Partitions**: 8 per experiment (for parallel execution)
- **Trials**: 512 per seed (64 per partition)
- **Total Trials**: 563,200 across all experiments

## Scripts Breakdown

### PPO (11 environments)
**Classic Control (5):**
- `ppo_cc_acrobot_partitioned.sh`
- `ppo_cc_cartpole_partitioned.sh`
- `ppo_cc_continuous_mountain_car_partitioned.sh`
- `ppo_cc_mountain_car_partitioned.sh`
- `ppo_cc_pendulum_partitioned.sh`

**Brax/MuJoCo (2):**
- `ppo_brax_fast_partitioned.sh`
- `ppo_brax_halfcheetah_partitioned.sh`

**MiniGrid/XLand (4):**
- `ppo_minigrid_door_key_partitioned.sh`
- `ppo_xland_door_key_partitioned.sh`
- `ppo_xland_empty_random_partitioned.sh`
- `ppo_xland_four_rooms_partitioned.sh`

### SAC (4 environments - continuous actions only)
**Classic Control (2):**
- `sac_cc_continuous_mountain_car_partitioned.sh`
- `sac_cc_pendulum_partitioned.sh`

**Brax/MuJoCo (2):**
- `sac_brax_fast_partitioned.sh`
- `sac_brax_halfcheetah_partitioned.sh`

### DQN (7 environments - discrete actions only)
**Classic Control (3):**
- `dqn_cc_acrobot_partitioned.sh`
- `dqn_cc_cartpole_partitioned.sh`
- `dqn_cc_mountain_car_partitioned.sh`

**MiniGrid/XLand (4):**
- `dqn_minigrid_door_key_partitioned.sh`
- `dqn_xland_door_key_partitioned.sh`
- `dqn_xland_empty_random_partitioned.sh`
- `dqn_xland_four_rooms_partitioned.sh`

## Usage

### 1. Submit All Jobs

```bash
cd experiment/kisski_cluster_scripts
bash submit_all_jobs.sh
```

This will submit all 22 experiments (176 parallel array jobs total).

### 2. Monitor Job Status

```bash
# View all your jobs
squeue -u $USER

# View specific algorithm jobs
squeue -u $USER | grep ppo
squeue -u $USER | grep sac
squeue -u $USER | grep dqn

# Count running jobs
squeue -u $USER | wc -l
```

### 3. Check Logs

```bash
# List error logs
ls -lh log/*.err

# Check for errors
tail log/*.err

# View specific job log
tail log/ppo_cc_acrobot_partitioned_*.out
```

### 4. Merge Results (After All Jobs Complete)

```bash
cd experiment/kisski_cluster_scripts
bash merge_all_experiments.sh
```

This will merge all 8 partitions for each of the 50 seeds across all 22 experiments.

### 5. Submit Individual Jobs

If you need to submit a single experiment:

```bash
cd experiment/kisski_cluster_scripts
sbatch ppo_cc_acrobot_partitioned.sh
```

## Job Configuration

Each script uses these SLURM parameters:
- `--array=0-7` (8 parallel partitions)
- `--cpus-per-task=2`
- `--gres=gpu:1`
- `--mem=64GB`
- `-t 1-12:00:00` (max 36 hours)

## Output Structure

Results are saved to:
```
multirun/{algorithm}_{environment}/{seed}/partition_{id}_of_8/
```

After merging:
```
multirun/{algorithm}_{environment}/{seed}/merged/
```

## Expected Compute Time

- Per partition: ~12-36 hours (varies by environment)
- Total wall time: ~12-36 hours (partitions run in parallel)
- Total CPU hours: 22 experiments × 8 partitions × 12-36 hours = ~2,112 - 6,336 CPU hours

## Troubleshooting

### Job Failed
```bash
# Check error log
tail log/{job_name}_partitioned_*.err

# Check output log
tail log/{job_name}_partitioned_*.out

# Resubmit specific job
sbatch {job_name}_partitioned.sh
```

### Out of Memory
- Increase `--mem=` in the script
- Reduce `n_trials` or number of seeds

### Partition Missing
- Check if partition job completed successfully
- Look for error in log file
- Re-run specific partition by modifying array index temporarily

## Files

- `submit_all_jobs.sh` - Master submission script
- `merge_all_experiments.sh` - Merge all partitioned results
- `log/` - Directory for SLURM output/error logs
- `*_partitioned.sh` - Individual experiment scripts

## Notes

- All scripts use `$HOME/Repos/arlbench` as the working directory
- Conda environment `arlbench` is activated automatically
- Mail notifications go to `m.che@ai.uni-hannover.de`
- Sobol seed is fixed at 42 for reproducibility
