"""Automated Reinforcement Learning Environment."""
from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from typing import Any

import gymnasium
import jax
import jax.numpy as jnp
import jax.tree_util
import numpy as np
import pandas as pd
from ConfigSpace import Configuration, ConfigurationSpace

from arlbench.core.algorithms import (
    DQN,
    PPO,
    SAC,
    Algorithm,
    AlgorithmState,
    TrainResult,
    TrainReturnT,
)
from arlbench.core.environments import make_env
from arlbench.utils import config_space_to_gymnasium_space

from .checkpointing import Checkpointer
from .objectives import OBJECTIVES, Objective
from .state_features import STATE_FEATURES, StateFeature

ObservationT = dict[str, np.ndarray]
ObjectivesT = dict[str, float]
InfoT = dict[str, Any]

DEFAULT_AUTO_RL_CONFIG = {
    "seed": 42,
    "env_framework": "gymnax",
    "env_name": "CartPole-v1",
    "env_kwargs": {},
    "eval_env_kwargs": {},
    "n_envs": 10,
    "algorithm": "dqn",
    "cnn_policy": False,
    "deterministic_eval": True,
    "nas_config": {},
    "checkpoint": [],
    "checkpoint_name": "default_checkpoint",
    "checkpoint_dir": "/tmp",
    "objectives": ["reward_mean"],
    "optimize_objectives": "upper",
    "state_features": [],
    "n_steps": 10,
    "n_total_timesteps": 1e5,
    "n_eval_steps": 100,
    "n_eval_episodes": 10,
}


class AutoRLEnv(gymnasium.Env):
    """Automated Reinforcement Learning (gynmasium-like) Environment.

    With each reset, the algorithm state is (re-)initialized.
    If a checkpoint path is passed to reset, the agent state is initialized with the checkpointed state.

    In each step, one iteration of training is performed with the current hyperparameter configuration (= action).
    """

    ALGORITHMS = {"ppo": PPO, "dqn": DQN, "sac": SAC}
    _algorithm: Algorithm
    _get_obs: Callable[[], np.ndarray]
    _algorithm_state: AlgorithmState | None
    _train_result: TrainResult | None
    _hpo_config: Configuration
    _total_training_steps: int

    def __init__(self, config: dict | None = None) -> None:
        """Creates a new AutoRL environment instance.

        Args:
            config (dict | None, optional): Configuration containing keys of DEFAULT_AUTO_RL_CONFIG.
            If no configuration keys are provided, default configuration is used. Defaults to None.
        """
        super().__init__()

        self._config = DEFAULT_AUTO_RL_CONFIG.copy()

        if config:
            for k, v in config.items():
                if k in DEFAULT_AUTO_RL_CONFIG:
                    self._config[k] = v
                else:
                    warnings.warn(
                        f"Invalid config key '{k}'. This item will be ignored."
                    )

        self._seed = int(self._config["seed"])

        self._done = True
        self._total_training_steps = 0  # timesteps across calls of step()
        self._c_step = 0  # current step
        self._c_episode = 0  # current episode

        # Environment
        self._env = make_env(
            self._config["env_framework"],
            self._config["env_name"],
            n_envs=self._config["n_envs"],
            env_kwargs=self._config["env_kwargs"],
            cnn_policy=self._config["cnn_policy"],
            seed=self._seed,
        )

        self._eval_env = make_env(
            self._config["env_framework"],
            self._config["env_name"],
            n_envs=self._config["n_envs"],
            env_kwargs=self._config["eval_env_kwargs"],
            cnn_policy=self._config["cnn_policy"],
            seed=self._seed + 1,
        )

        # Checkpointing
        self._checkpoints = []
        self._track_metrics = (
            "all" in self._config["checkpoint"]
            or "loss" in self._config["checkpoint"]
            or "grad_info" in self._config["state_features"]
            or "loss_info" in self._config["state_features"]
        )
        self._track_trajectories = (
            "all" in self._config["checkpoint"]
            or "trajectories" in self._config["checkpoint"]
            or "prediction_info" in self._config["state_features"]
            or any(["train_reward" in o for o in self._config["objectives"]])
        )

        # Algorithm
        self._algorithm_cls = self.ALGORITHMS[self._config["algorithm"]]
        self._config_space = self._algorithm_cls.get_hpo_config_space()

        # Instantiate algorithm with default hyperparameter configuration
        self._nas_config = self._algorithm_cls.get_default_nas_config()
        for k, v in self._config["nas_config"].items():
            self._nas_config[k] = v
        self._hpo_config = self._algorithm_cls.get_default_hpo_config()
        self._algorithm = self._make_algorithm()
        self._algorithm_state = None
        self._train_result = None

        # Optimization objectives
        self._objectives, self._objective_arguments = self._get_objectives()

        # State Features
        self._state_features = self._get_state_features()

        self._observation_space = self._get_obs_space()

    def _get_objectives(self) -> list[Objective]:
        """Maps the objectives as list of strings to a sorted list of the actual objective classes.

        Returns:
            list[Objective]: List of objective classes in the correct order to be wrapped around the train function.
        """
        if len(self._config["objectives"]) == 0:
            raise ValueError("Please select at least one optimization objective.")

        objectives = []
        objective_arguments = []
        cfg_objectives = list(set(self._config["objectives"]))

        for o in cfg_objectives:
            if o not in OBJECTIVES:
                matching_base = None
                for base in OBJECTIVES:
                    if o.startswith(base + "_"):
                        matching_base = base
                        break
                if matching_base:
                    parts = o.split("_")
                    base_parts = matching_base.split("_")
                    args = parts[len(base_parts):]
                    base_o = matching_base
                    arg_dict = {}

                    if len(args) % 2 != 0:
                        arg_dict["default_arg"] = args[0]
                        args = args[1:]

                    for i in range(0, len(args), 2):
                        arg_dict[args[i]] = float(args[i + 1])

                    objectives.append(OBJECTIVES[base_o])
                    objective_arguments.append(arg_dict)
                else:
                    raise ValueError(f"Invalid objective: {o}")
            else:
                objectives.append(OBJECTIVES[o])
                objective_arguments.append({})

        # Ensure right order of objectives, e.g. runtime is wrapped first
        objective_arguments = [args for _, args in sorted(zip(objectives, objective_arguments, strict=False), key=lambda o: o[0][1])]
        objectives = sorted(objectives, key=lambda o: o[1])

        # Now we are extracting the actual classes for each objective
        # They are used to wrap the train function and compute the objective
        objectives = [o[0] for o in objectives]
        return objectives, objective_arguments

    def _get_state_features(self) -> list[StateFeature]:
        """Maps the state features as list of strings to a sorted list of the actual state feature classes.

        Returns:
            list[StateFeature]: List of state features classes in the correct order to be wrapped around the train function.
        """
        state_features = []
        cfg_state_features = list(set(self._config["state_features"]))
        for f in cfg_state_features:
            if f not in STATE_FEATURES:
                raise ValueError(f"Invalid state feature: {f}")
            state_features += [STATE_FEATURES[f]]
        return state_features

    def _get_obs_space(self) -> gymnasium.spaces.Dict:
        """Returns the state feature space as gymnasium space.

        Returns:
            gymnasium.spaces.Dict: Gymnasium space.
        """
        obs_space = {f.KEY: f.get_state_space(self._config["algorithm"]) for f in self._state_features}

        obs_space["steps"] = gymnasium.spaces.Box(
            low=np.array([0, 0]), high=np.array([np.inf, np.inf])
        )

        return gymnasium.spaces.Dict(obs_space)

    def _step(self) -> bool:
        """Fundamental step behaviour, handles truncation.

        Returns:
            bool: Whether the episode is done.
        """
        self._c_step += 1
        if self._c_step >= self._config["n_steps"]:
            return True
        return False

    def _train(self, **train_kw_args) -> tuple[TrainReturnT, dict, dict]:
        """Performs the RL training and returns the result as well as objectives and state features.

        Returns:
            tuple[TrainReturnT, dict, dict]: Tuple of training result, objectives, and observation (state features).
        """
        assert self._algorithm_state is not None

        objectives = {}  # result are stored here
        obs = {}  # state features are stored here
        train_func = self._algorithm.train

        # The objectives are wrapped first since runtime should be accurate
        for o, args in zip(self._objectives, self._objective_arguments, strict=False):
            train_func = o(train_func, objectives, self._config["optimize_objectives"], **args)

        # Then we wrap the state features around the training function
        obs["steps"] = np.array([self._c_step, self._total_training_steps])
        for f in self._state_features:
            train_func = f(train_func, obs)

        # Track configuration + budgets using deepcave (https://github.com/automl/DeepCAVE)
        if self._config.get("deep_cave", False):
            from deepcave import Objective, Recorder

            dc_objectives = [Objective(**o.get_spec()) for o in self._objectives]

            with Recorder(self._config_space, objectives=dc_objectives) as r:
                r.start(self._hpo_config, self._config["n_total_timesteps"])
                result = train_func(*self._algorithm_state, **train_kw_args)

                r.end(costs=[objectives[o.KEY] for o in self._objectives])
        else:
            result = train_func(*self._algorithm_state, **train_kw_args)
        return result, objectives, obs

    def _make_algorithm(self) -> Algorithm:
        """Instantiated the RL algorithm given the current AutoRL config and hyperparameter configuration.

        Returns:
            Algorithm: RL algorithm instance.
        """
        return self._algorithm_cls(
            self._hpo_config,
            self._env,
            nas_config=self._nas_config,
            eval_env=self._eval_env,
            track_metrics=self._track_metrics,
            track_trajectories=self._track_trajectories,
            cnn_policy=self._config["cnn_policy"],
            deterministic_eval=self._config["deterministic_eval"],
        )

    def get_algorithm_init_kwargs(self, init_rng) -> dict:
        """Returns the algorithm initialization parameters.

        Returns:
            Dict: Dictionary of algorithm initialization parameters.
        """
        if isinstance(self._algorithm, PPO):
            return {"rng": init_rng, "network_params": self._algorithm_state.runner_state.train_state.params, "opt_state": self._algorithm_state.runner_state.train_state.opt_state}
        elif isinstance(self._algorithm, DQN):
            return{
                    "rng": init_rng,
                    "buffer_state": self._algorithm_state.buffer_state,
                    "network_params": self._algorithm_state.runner_state.train_state.params,
                    "target_params": self._algorithm_state.runner_state.train_state.target_params,
                    "opt_state": self._algorithm_state.runner_state.train_state.opt_state,
                }
        elif isinstance(self._algorithm, SAC):
            return {
                    "rng": init_rng,
                    "buffer_state": self._algorithm_state.buffer_state,
                    "actor_network_params": self._algorithm_state.runner_state.actor_train_state.params,
                    "critic_network_params": self._algorithm_state.runner_state.critic_train_state.params,
                    "critic_target_params": self._algorithm_state.runner_state.critic_train_state.target_params,
                    "alpha_network_params": self._algorithm_state.runner_state.alpha_train_state.params,
                    "actor_opt_state": self._algorithm_state.runner_state.actor_train_state.opt_state,
                    "critic_opt_state": self._algorithm_state.runner_state.critic_train_state.opt_state,
                    "alpha_opt_state": self._algorithm_state.runner_state.alpha_train_state.opt_state,
                }
        else:
            raise ValueError(f"Unsupported algorithm: {self._algorithm.name}")

    def _step_multi_seed(
        self,
        action: Configuration | dict,
        checkpoint_path: str | None,
        n_total_timesteps: int | None,
        n_eval_steps: int | None,
        n_eval_episodes: int | None,
        seeds: list[int],
    ) -> tuple[ObservationT, ObjectivesT, bool, bool, InfoT]:
        """Performs one iteration of RL training for multiple seeds simultaneously."""
        self._algorithm = self._make_algorithm()

        # Initialize or load states for all seeds
        if self._algorithm_state is None:
            if checkpoint_path is None:
                rngs = jnp.stack([jax.random.key(s) for s in seeds])
                self._algorithm_state = jax.vmap(self._algorithm.init)(rngs)
            else:
                # Initialize for all seeds
                states = []
                for s in seeds:
                    if checkpoint_path:
                        try:
                            state = self._load(checkpoint_path, s)
                        except Exception as e:  # noqa: BLE001
                            print(e)
                            init_rng = jax.random.key(s)
                            state = self._algorithm.init(init_rng)
                    else:
                        init_rng = jax.random.key(s)
                        state = self._algorithm.init(init_rng)
                    states.append(state)

                # Stack states
                self._algorithm_state = jax.tree_util.tree_map(
                    lambda *args: jnp.stack(args), *states
                )
        else:
            # Re-initialize for each seed, preserving weights from current state
            # This handles the transition from Single -> Multi seed
            dummy_rng = jax.random.key(0)
            # Get current weights/state
            base_kwargs = self.get_algorithm_init_kwargs(dummy_rng)
            if "rng" in base_kwargs:
                del base_kwargs["rng"]

            # Create batch of RNGs
            rngs = jnp.stack([jax.random.key(s) for s in seeds])

            def init_fn(rng):
                return self._algorithm.init(rng, **base_kwargs)

            self._algorithm_state = jax.vmap(init_fn)(rngs)

        # Training kwargs
        n_total_timesteps = (
            n_total_timesteps
            if n_total_timesteps
            else self._config["n_total_timesteps"]
        )
        n_eval_steps = (
            n_eval_steps if n_eval_steps else self._config["n_eval_steps"]
        )
        n_eval_episodes = (
            n_eval_episodes if n_eval_episodes else self._config["n_eval_episodes"]
        )

        vmap_train = jax.vmap(self._algorithm.train, in_axes=(0, 0, None, None, None))

        start_time = time.time()

        batched_state, batched_result = vmap_train(
            *self._algorithm_state, n_total_timesteps, n_eval_steps, n_eval_episodes
        )

        runtime = time.time() - start_time

        self._algorithm_state = batched_state
        self._train_result = batched_result  # This is now batched

        # Compute objectives
        objectives = {}

        # Runtime
        if "runtime" in self._config["objectives"]:
            # Runtime is shared/total for the batch
            objectives["runtime"] = runtime
            if self._config["optimize_objectives"] != "lower":  # Runtime is naturally lower
                objectives["runtime"] *= -1

        # Rewards
        eval_rewards = batched_result.eval_rewards
        last_eval_rewards = eval_rewards[:, -1, :]  # (n_seeds, n_eval_episodes)

        for o_name in self._config["objectives"]:
            if o_name.startswith("reward"):
                # e.g. reward_mean
                parts = o_name.split("_")
                agg = parts[1] if len(parts) > 1 else "mean"

                # Aggregate over episodes
                episode_agg = getattr(np, agg)(last_eval_rewards, axis=1)  # (n_seeds,)

                # Aggregate over seeds (always mean?)
                seed_agg = np.mean(episode_agg)

                val = seed_agg.item()
                if (
                    self._config["optimize_objectives"] == "lower"
                ):  # Reward is naturally upper
                    val *= -1
                objectives[o_name] = val

        # Observations
        obs = {}
        obs["steps"] = np.array([self._c_step, self._total_training_steps])

        self._total_training_steps += n_total_timesteps

        info = {}
        steps = (
            np.arange(1, n_eval_steps + 1)
            * n_total_timesteps
            // n_eval_steps
        )
        # returns: mean over seeds and episodes
        returns = eval_rewards.mean(axis=(0, 2))
        info["train_info_df"] = pd.DataFrame({"steps": steps, "returns": returns})

        return obs, objectives, False, self._done, info

    def step(
        self,
        action: Configuration | dict,
        checkpoint_path: str | None = None,
        n_total_timesteps: int | None = None,
        n_eval_steps: int | None = None,
        n_eval_episodes: int | None = None,
        seed: int | list[int] | None = None,
    ) -> tuple[ObservationT, ObjectivesT, bool, bool, InfoT]:
        """Performs one iteration of RL training.

        Args:
            action (Configuration | dict): Hyperparameter configuration to use for training.
            n_total_timesteps (int | None, optional): Number of total training steps. Defaults to None.
            n_eval_steps (int | None, optional): Number of evaluations during training. Defaults to None.
            n_eval_episodes (int | None, optional): Number of episodes to run per evalution during training. Defaults to None.
            seed (int | list[int] | None, optional): Random seed(s). Defaults to None. If None, seed of the AutoRL environment is used.

        Raises:
            ValueError: Error is raised if step() is called before reset() was called.

        Returns:
            tuple[ObservationT, ObjectivesT, bool, bool, InfoT]: State information, objectives, terminated, truncated, additional information.
        """
        if len(action.keys()) == 0:  # no action provided
            warnings.warn(
                "No agent configuration provided. Falling back to default configuration."
            )

        if self._done:
            raise ValueError("Called step() before reset().")

        # Set done if max. number of steps in DAC is reached or classic (one-step) HPO is performed
        self._done = self._step()
        info = {}

        # Apply changes to current hyperparameter configuration and reinstantiate algorithm
        if isinstance(action, dict):
            action_config = dict(self._hpo_config)
            action_config.update(action)
            action = Configuration(self.config_space, action_config)
        self._hpo_config = action

        seed = seed if seed else self._seed

        if isinstance(seed, list):
            return self._step_multi_seed(
                action,
                checkpoint_path,
                n_total_timesteps,
                n_eval_steps,
                n_eval_episodes,
                seed,
            )

        self._algorithm = self._make_algorithm()

        # First, we check if there is a checkpoint to load. If not, we have to check
        # whether this is the first iteration, i.e., call of env.step(). In that case,
        # we have to initialiaze the algorithm state.
        # Otherwise, we are using the state from previous iteration(s)
        if checkpoint_path:
            try:
                self._algorithm_state = self._load(checkpoint_path, seed)
            except Exception as e: # noqa: BLE001
                print(e)
                init_rng = jax.random.key(seed)
                self._algorithm_state = self._algorithm.init(init_rng)
        elif self._algorithm_state is None:
            init_rng = jax.random.key(seed)
            self._algorithm_state = self._algorithm.init(init_rng)
        else:
            init_rng = jax.random.key(seed)
            init_kwargs = self.get_algorithm_init_kwargs(init_rng)
            self._algorithm_state = self._algorithm.init(**init_kwargs)

        # Training kwargs
        train_kw_args = {
            "n_total_timesteps": n_total_timesteps
            if n_total_timesteps
            else self._config["n_total_timesteps"],
            "n_eval_steps": n_eval_steps
            if n_eval_steps
            else self._config["n_eval_steps"],
            "n_eval_episodes": n_eval_episodes
            if n_eval_episodes
            else self._config["n_eval_episodes"],
        }

        # Perform one iteration of training
        result, objectives, obs = self._train(**train_kw_args)
        self._algorithm_state, self._train_result = result

        steps = (
            np.arange(1, train_kw_args["n_eval_steps"] + 1)
            * train_kw_args["n_total_timesteps"]
            // train_kw_args["n_eval_steps"]
        )
        returns = self._train_result.eval_rewards.mean(axis=1)

        info["train_info_df"] = pd.DataFrame({"steps": steps, "returns": returns})

        self._total_training_steps += train_kw_args["n_total_timesteps"]

        # Checkpointing
        if len(self._config["checkpoint"]) > 0:
            assert self._algorithm_state is not None

            checkpoint = self._save()
            self._checkpoints += [checkpoint]
            info["checkpoint"] = checkpoint

        return obs, objectives, False, self._done, info

    def reset(
        self
    ) -> tuple[ObservationT, InfoT]:
        """Resets the AutoRL environment and current algorithm state.

        Returns:
            tuple[ObservationT, InfoT]: Empty observation and state information.
        """
        self._done = False
        self._c_step = 0
        self._c_episode += 1
        self._algorithm_state = None

        return {}, {}

    def _save(self, tag: str | None = None) -> str:
        """Saves the current algorithm state and training result.

        Args:
            tag (str | None, optional): Checkpoint tag. Defaults to None.

        Returns:
            str: Checkpoint path.
        """
        if self._algorithm_state is None:
            warnings.warn("Agent not initialized. Not able to save agent state.")
            return ""

        if self._train_result is None:
            warnings.warn(
                "No training performed, so there is nothing to save. Please run step() first."
            )

        return Checkpointer.save(
            self._algorithm.name,
            self._algorithm_state,
            self._config,
            self._hpo_config,
            self._done,
            self._c_episode,
            self._c_step,
            self._train_result,
            tag=tag,
        )

    def _load(self, checkpoint_path: str, seed: int) -> AlgorithmState:
        """Load the algorithm state from a checkpoint.

        Args:
            checkpoint_path (str): Path of the checkpoint to load.
            seed (int | None, optional): Random seed to use for algorithm initialization. Defaults to None.
        """
        init_rng = jax.random.PRNGKey(seed)
        algorithm_state = self._algorithm.init(init_rng)
        (
            (
                _,
                self._c_step,
                self._c_episode,
            ),
            algorithm_kw_args,
        ) = Checkpointer.load(checkpoint_path, algorithm_state)
        return self._algorithm.init(init_rng, **algorithm_kw_args)

    @property
    def action_space(self) -> gymnasium.spaces.Space:
        """Returns the hyperparameter configuration spaces as gymnasium space.

        Returns:
            gymnasium.spaces.Space: Hyperparameter configuration space.
        """
        return config_space_to_gymnasium_space(self._config_space)

    @property
    def config_space(self) -> ConfigurationSpace:
        """Returns the hyperparameter configuration spaces as ConfigSpace.

        Returns:
            ConfigurationSpace: Hyperparameter configuration space.
        """
        return self._config_space

    @property
    def observation_space(self) -> gymnasium.spaces.Space:
        """Returns a gymnasium spaces of state features (observations).

        Returns:
            gymnasium.spaces.Space: Gynasium space.
        """
        return self._observation_space

    @property
    def hpo_config(self) -> Configuration:
        """Returns the current hyperparameter configuration stored in the AutoRL environment..

        Returns:
            Configuration: Hyperparameter configuration.
        """
        return self._hpo_config

    @property
    def checkpoints(self) -> list[str]:
        """Returns a list of created checkpoints for this AutoRL environment.

        Returns:
            list[str]: List of checkpoint paths.
        """
        return list(self._checkpoints)

    @property
    def objectives(self) -> list[str]:
        """Returns configured objectives.

        Returns:
            list[str]: List of objectives.
        """
        return [o.__name__ for o in self._objectives]

    @property
    def config(self) -> dict:
        """Returns the AutoRL configuration.

        Returns:
            dict: AutoRL configuration.
        """
        return self._config.copy()

    def eval(self, num_eval_episodes: int) -> np.ndarray:
        """Evaluates the algorithm using its current training state.

        Args:
            num_eval_episodes (int): Number of evaluation episodes to run.

        Returns:
            np.ndarray: Array of evaluation return for each episode.
        """
        if self._algorithm is None or self._algorithm_state is None:
            raise ValueError("Agent not initialized. Call reset() first.")
        rewards = self._algorithm.eval(
            self._algorithm_state.runner_state, num_eval_episodes
        )
        return np.array(rewards)
