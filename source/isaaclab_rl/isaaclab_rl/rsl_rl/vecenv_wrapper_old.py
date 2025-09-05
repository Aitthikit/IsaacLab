# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import torch

from rsl_rl.env import VecEnv

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

import copy
import gym
import gym.spaces
import os
import torch
from typing import Dict, Optional, Tuple

# rsl-rl
# from rsl_rl.env.vec_env import VecEnv

# from omni.isaac.orbit_envs.isaac_env import IsaacEnv

__all__ = ["RslRlVecEnvWrapper"]


"""
Vectorized environment wrapper.
"""

# VecEnvObs is what is returned by the reset() method
# it contains the observation for each env
VecEnvObs = Tuple[torch.Tensor, Optional[torch.Tensor]]
# VecEnvStepReturn is what is returned by the step() method
# it contains the observation (actor and critic), reward, done, info for each env
VecEnvStepReturn = Tuple[VecEnvObs, VecEnvObs, torch.Tensor, torch.Tensor, Dict]


class RslRlVecEnvWrapper(gym.Wrapper, VecEnv):
    """Wraps around Isaac Orbit environment for RSL-RL.

    To use asymmetric actor-critic, the environment instance must have the attributes :attr:`num_states` (int)
    and :attr:`state_space` (:obj:`gym.spaces.Box`). These are used by the learning agent to allocate buffers in
    the trajectory memory. Additionally, the method :meth:`_get_observations()` should have the key "critic"
    which corresponds to the privileged observations. Since this is optional for some environments, the wrapper
    checks if these attributes exist. If they don't then the wrapper defaults to zero as number of privileged
    observations.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self, env: DirectRLEnv | ManagerBasedRLEnv):
        """Initializes the wrapper.

        Args:
            env (IsaacEnv): The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`IsaacEnv`.
            ValueError: When the observation space is not a :obj:`gym.spaces.Box`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(f"The environment must be inherited from IsaacEnv. Environment type: {type(env)}")
        # initialize the wrapper
        gym.Wrapper.__init__(self, env)
        # check that environment only provides flatted obs
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(
                f"RSL-RL only supports flattened observation spaces. Input observation space: {env.observation_space}"
            )
        # store information required by wrapper
        self.num_envs = self.env.unwrapped.num_envs
        self.num_actions = self.env.action_space.shape[0]
        self.num_obs = self.env.observation_space.shape[0]
        # information for privileged observations
        self.privileged_obs_space = getattr(self.env, "state_space", None)
        self.num_privileged_obs = getattr(self.env, "num_states", None)

    """
    Properties
    """

    def get_observations(self) -> torch.Tensor:
        """Returns the current observations of the environment."""
        return self.env.unwrapped._get_observations()["policy"]

    def get_privileged_observations(self) -> Optional[torch.Tensor]:
        """Returns the current privileged observations of the environment (if available)."""
        if self.num_privileged_obs is not None:
            try:
                privileged_obs = self.env.unwrapped._get_observations()["critic"]
            except AttributeError:
                raise NotImplementedError("Environment does not define the key `critic` for privileged observations.")
        else:
            privileged_obs = None

        return privileged_obs

    """
    Operations - MDP
    """

    def reset(self) -> VecEnvObs:  # noqa: D102
        # reset the environment
        obs_dict = self.env.reset()
        # return observations
        return self._process_obs(obs_dict)

    def step(self, actions: torch.Tensor) -> VecEnvStepReturn:  # noqa: D102
        # record step information
        obs_dict, rew, dones, extras = self.env.step(actions)
        # process observations
        obs, privileged_obs = self._process_obs(obs_dict)
        # return step information
        return obs, privileged_obs, rew, dones, extras

    """
    Helper functions
    """

    def _process_obs(self, obs_dict: dict) -> VecEnvObs:
        """Processing of the observations from the environment.

        Args:
            obs (dict): The current observations from environment.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: The observations for actor and critic. If no
                privileged observations are available then the critic observations are set to :obj:`None`.
        """
        # process policy obs
        obs = obs_dict["policy"]
        # process critic observations
        # note: if None then policy observations are used
        if self.num_privileged_obs is not None:
            try:
                privileged_obs = obs_dict["critic"]
            except AttributeError:
                raise NotImplementedError("Environment does not define the key `critic` for privileged observations.")
        else:
            privileged_obs = None
        # return observations
        return obs, privileged_obs


"""
Helper functions.
"""
