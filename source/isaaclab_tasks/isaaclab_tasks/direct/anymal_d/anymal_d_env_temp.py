# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor , RayCaster
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.markers import VisualizationMarkers
import random
import math

from .anymal_d_env_cfg import AnymalDFlatEnvCfg, AnymalDRoughEnvCfg, AnymalDClimbEnvCfg , AnymalDClimbUpEnvPosCfg , AnymalDClimbDownEnvPosCfg, AnymalDFlatEnvPosCfg


class AnymalDEnv(DirectRLEnv):
    cfg: AnymalDFlatEnvCfg | AnymalDRoughEnvCfg | AnymalDClimbEnvCfg

    def __init__(self, cfg: AnymalDFlatEnvCfg | AnymalDRoughEnvCfg | AnymalDClimbEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        # print(torch.zeros(1,3,self.device)
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, AnymalDClimbEnvCfg) or isinstance(self.cfg, AnymalDRoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None
        # if isinstance(self.cfg, AnymalDRoughEnvCfg):
        if isinstance(self.cfg, AnymalDClimbEnvCfg) or isinstance(self.cfg, AnymalDRoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
        # print(height_data)
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    height_data,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        robot_height = torch.sum(torch.square(self._robot.data.body_com_pos_w[0, 0, 2]))
        # print(self._commands[:, :3])
        # print(self.episode_length_buf)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            # "robot_height": robot_height * self.cfg.robot_height_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = (self.episode_length_buf >= self.max_episode_length - 1)
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        print(self._commands)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)





class AnymalDEnvPos(DirectRLEnv):
    cfg: AnymalDFlatEnvPosCfg | AnymalDClimbUpEnvPosCfg | AnymalDClimbDownEnvPosCfg

    def __init__(self, cfg: AnymalDFlatEnvPosCfg | AnymalDClimbUpEnvPosCfg | AnymalDClimbDownEnvPosCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X Y Z position and Time remaining
        self._commands = torch.zeros(self.num_envs, 4, device=self.device)
        self.reward_task_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
        marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
        self.goal_visualizer = VisualizationMarkers(marker_cfg)
        self.goal_visualizer.set_visibility(True)
        #######################################################################################################################################
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "dof_vel_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "task_reward",
                "bias_reward",
                "stall_reward",
                "feet_accel",
                "heading",
                "termination",
                "feet_contact_force",
                "base_accel",
                "stumble"
            ]
        }

        ########################################################################################################################################
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")
        self._shank_ids,_ = self._contact_sensor.find_bodies(".*SHANK")
        

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        # if isinstance(self.cfg, AnymalDClimbUpEnvPosCfg) or isinstance(self.cfg, AnymalDClimbDownEnvPosCfg) or isinstance(self.cfg, AnymalDClimbDownEnvPosCfg):
        # we add a height scanner for perceptive locomotion
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        ################################################################################################################
        self._previous_actions = self._actions.clone()
        height_data = None
        # if  isinstance(self.cfg, AnymalDClimbUpEnvPosCfg) or isinstance(self.cfg, AnymalDClimbDownEnvPosCfg) or isinstance(self.cfg, AnymalDClimbDownEnvPosCfg):
        # # if isinstance(self.cfg, AnymalDClimbEnvCfg):
        height_data = (
            self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
        ).clip(-1.0, 1.0)
        # print(height_data)
        commandobs = torch.clone(self._commands)
        commandobs[:, :2] = self._commands[:, :2]-self._robot.data.root_pos_w[:, :2]
        commandobs[:,3] = (self._commands[:,3] - self.episode_length_buf)/500
        # print(commandobs[:,3])
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    commandobs,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    height_data,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations
        ################################################################################################################
    def _get_rewards(self) -> torch.Tensor:
        ################################################################################################################
        # print(self._contact_sensor.find_bodies(".*THIGH"))
        # linear velocity tracking
        # lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        # lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # # yaw rate tracking
        # yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        # yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # # z velocity tracking
        # z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # # angular velocity x/y
        # ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        # joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        joint_velo = torch.sum(torch.square(self._robot.data.joint_vel), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # base acceleration
        base_accel = (torch.sum(torch.square(torch.norm(self._robot.data.body_lin_acc_w,dim=1)),dim=1) + 
                    (0.02 * torch.sum(torch.square(torch.norm(self._robot.data.body_ang_acc_w,dim=1)),dim=1)))
        # print(f"robot_acc : {base_accel}")
        # print(f"robot_ang_acc : {self._robot.data.body_ang_acc_w}")        
        # feet air time
        # first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        # last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        # air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
        #     torch.norm(self._commands[:, :2], dim=1) > 0.1
        # )
        desired_heading = torch.atan2(self._commands[:, 1] - self._robot.data.root_pos_w[:, 1], self._commands[:, 0] - self._robot.data.root_pos_w[:, 0])
        # print(f'target{desired_heading}')
        yaw = torch.atan2(2.0 * (self._robot.data.body_com_quat_w[:,self._base_id,3] * 
                                 self._robot.data.body_com_quat_w[:,self._base_id,2] + 
                                 self._robot.data.body_com_quat_w[:,self._base_id,0] * 
                                 self._robot.data.body_com_quat_w[:,self._base_id,1]),
                                 1.0 - 2.0 * 
                                 (self._robot.data.body_com_quat_w[:,self._base_id,1] * 
                                    self._robot.data.body_com_quat_w[:,self._base_id,1] + 
                                    self._robot.data.body_com_quat_w[:,self._base_id,2] * 
                                    self._robot.data.body_com_quat_w[:,self._base_id,2])).squeeze()  # You need to compute this
        # print(f"robot{yaw}")
        r_heading = torch.cos(yaw - desired_heading)
        # print(f"error{r_heading}")
       


        feet_accel = torch.sum(torch.square(torch.norm(self._robot.data.body_lin_acc_w[:,13:17],dim=1)),dim=1)

        # undesired contacts
        if isinstance(self.cfg, AnymalDClimbUpEnvPosCfg):
            net_contact_forces = self._contact_sensor.data.net_forces_w_history
            is_contact = (
                torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0
            )
            contacts = torch.sum(is_contact, dim=1)
        else:
            net_contact_forces = self._contact_sensor.data.net_forces_w_history
            is_contact = torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
            is_contact2 = torch.max(torch.norm(net_contact_forces[:, :, self._shank_ids], dim=-1), dim=1)[0] > 1.0
            contacts = torch.sum(is_contact, dim=1)
            contacts2 = torch.sum(is_contact2, dim=1)
            contacts += contacts2 

        # foot force penalty
        # foot_force_norms = torch.norm(net_contact_forces[:, :, self._feet_ids],dim=1)
        foot_force_norms = torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids],dim=1)
        feet_contact_force = torch.sum(torch.square(torch.maximum(foot_force_norms-700,torch.zeros_like(foot_force_norms))),dim=1)
        stumble_mask = torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids,:2],dim=-1) > self._contact_sensor.data.net_forces_w[:, self._feet_ids,2] * 2
        stumble_penalty = torch.where(stumble_mask, torch.full_like(stumble_mask, 1.0), torch.zeros_like(stumble_mask))
        stumble_penalty = torch.sum(stumble_penalty,dim=1)
        # Termination
            
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        # Compute time condition per environment
        time_condition = (self.episode_length_buf / 50) > (self.max_episode_length_s - 5)

        # Compute task reward per environment
        reward_task = torch.where(
            time_condition,
            0.2 * (1.0 / (1.0 + torch.square(torch.norm(self._robot.data.root_pos_w[:, :2] - self._commands[:, :2],dim=1)))),
            torch.zeros(self.num_envs, device=self.device)
        )
        self.reward_task_reached |= reward_task >= 0.15

        if (~self.reward_task_reached.all()):
            reward_bias = (
            torch.sum(self._robot.data.root_lin_vel_b[:, :3] * (self._commands[:, :3] - self._robot.data.root_pos_w[:, :3]), dim=1) /
            (torch.norm(self._robot.data.root_lin_vel_b[:, :3], dim=1) *
             torch.norm(self._commands[:, :3] - self._robot.data.root_pos_w[:, :3], dim=1))
            )
        else:
            reward_bias = torch.zeros(self.num_envs,device=self.device)
        # Compute per-env values
        velocity = torch.norm(self._robot.data.root_lin_vel_b[:, :3], dim=1)
        pos_error = torch.norm(self._commands[:, :3] - self._robot.data.root_pos_w[:, :3], dim=1)
        # Condition: low velocity AND high position error, per robot
        stall_mask = (velocity < 0.1) & (pos_error > 0.5)
        # Assign -1 reward where the condition is met
        reward_stall = torch.where(stall_mask, torch.full_like(velocity, 1.0), torch.zeros_like(velocity))

        rewards = {
            # "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            # "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            # "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            # "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            # "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "dof_vel_l2" : joint_velo * self.cfg.joint_velo_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "heading": r_heading * self.cfg.heading_reward_scale * self.step_dt,
            # "base_accel" : base_accel * self.cfg.base_accel_reward_scale * self.step_dt,
            # "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "feet_accel" : feet_accel * self.cfg.feet_accel_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            # "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "task_reward" : reward_task*self.cfg.task_reward_scale * self.step_dt,
            "bias_reward" : reward_bias*self.cfg.bias_reward_scale * self.step_dt,
            "stall_reward" : reward_stall*self.cfg.stall_reward_scale * self.step_dt,
            "termination" : self.cfg.terminate_reward_scale * self.reset_terminated.float(),
            "feet_contact_force" : self.cfg.feet_force_reward_scale * feet_contact_force * self.step_dt,
            "stumble" : self.cfg.stumble_reward_scale * stumble_penalty,

            # "robot_height": robot_height * self.cfg.robot_height_reward_scale * self.step_dt,
        }
        # if ((rewards["task_reward"] > 0.1).any()):
        #     self.reward_task_reach = False
        # print(rewards)
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward
        ################################################################################################################
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # reach = torch.norm(self._commands[:, :3] - self._robot.data.root_pos_w[:, :3], dim=1) <= 0.5
        # if  isinstance(self.cfg, AnymalDClimbUpEnvPosCfg):
        #     died = torch.zeros(self.num_envs,dtype=torch.bool)
        # else:
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = (torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1))
        died |= torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._feet_ids], dim=-1), dim=1)[0] > 1500.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
            self.episode_temp = torch.clone(self.episode_length_buf)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        ################################################################################################################
        # Sample new commands
        if  isinstance(self.cfg, AnymalDFlatEnvPosCfg):
            angle = torch.rand(1,device=self.device) * 2 * torch.pi
            radius = 1 + 4 * torch.rand(1,device=self.device)   
            self._commands[env_ids,0] = self.scene.env_origins[env_ids,0]
            self._commands[env_ids,1] = self.scene.env_origins[env_ids,1]
            self._commands[env_ids,0] += radius * torch.cos(angle)
            self._commands[env_ids,1] += radius * torch.sin(angle)
            self._commands[env_ids,2] = self._robot.data.default_root_state[env_ids, 2]
            self._commands[env_ids,3] = torch.tensor(self.max_episode_length,device=self.device,dtype=torch.float32)
            # (torch.tensor(self.max_episode_length,device=self.device,dtype=torch.float32) - self.episode_length_buf[env_ids].float())/20
            default_root_state = self._robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        if  isinstance(self.cfg, AnymalDClimbUpEnvPosCfg):
            self._commands[env_ids,0] = self._terrain.env_origins[env_ids,0]
            self._commands[env_ids,1] = self._terrain.env_origins[env_ids,1]
            angle = torch.rand(1,device=self.device) * 2 * torch.pi
            radius = torch.tensor(4.5,device=self.device)
            self._commands[env_ids, 0] += radius * torch.cos(angle)
            self._commands[env_ids, 1] += radius * torch.sin(angle)
            self._commands[env_ids, 2] = 0.6
            self._commands[env_ids, 3] = torch.tensor(self.max_episode_length,device=self.device,dtype=torch.float32)
            default_root_state = self._robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            # self._commands[env_ids,0] = self._terrain.env_origins[env_ids,0]
            # self._commands[env_ids,1] = self._terrain.env_origins[env_ids,1]
            # self._commands[env_ids, 2] = self._terrain.env_origins[env_ids,2] + 0.6
            # self._commands[env_ids, 3] = torch.tensor(self.max_episode_length,device=self.device,dtype=torch.float32)
            # angle = torch.rand(1,device=self.device) * 2 * torch.pi
            # radius = torch.tensor(4.5,device=self.device)   
            # default_root_state = self._robot.data.default_root_state[env_ids]
            # default_root_state[:, 0] += self._terrain.env_origins[env_ids, 0] + radius * torch.cos(angle)
            # default_root_state[:, 1] += self._terrain.env_origins[env_ids, 1] + radius * torch.sin(angle)
        if  isinstance(self.cfg, AnymalDClimbDownEnvPosCfg):
            self._commands[env_ids,0] = self._terrain.env_origins[env_ids,0]
            self._commands[env_ids,1] = self._terrain.env_origins[env_ids,1]
            angle = torch.rand(1,device=self.device) * 2 * torch.pi
            radius = torch.tensor(4.5,device=self.device)
            self._commands[env_ids, 0] += radius * torch.cos(angle)
            self._commands[env_ids, 1] += radius * torch.sin(angle)
            self._commands[env_ids, 2] = 0.6
            self._commands[env_ids, 3] = torch.tensor(self.max_episode_length,device=self.device,dtype=torch.float32)
            default_root_state = self._robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        # Get the scene origin position
        # self._commands[env_ids,0] = self.scene.env_origins[env_ids,0]
        # self._commands[env_ids,1] = self.scene.env_origins[env_ids,1]
        # self._commands[env_ids,0] = self._terrain.env_origins[env_ids,0]
        # self._commands[env_ids,1] = self._terrain.env_origins[env_ids,1]
        # offset the position command by the current root position
        # r = torch.empty(len(env_ids), device=self.device)
        # self._commands[env_ids, 0] += random.uniform(-3.0,3.0)
        # self._commands[env_ids, 1] += random.uniform(-3.0,3.0)
        # self._commands[env_ids, 2] = self._robot.data.default_root_state[env_ids, 2]
        # self._commands[env_ids, 3] = 5
        # angle = torch.rand(1,device=self.device) * 2 * torch.pi
        # radius = torch.tensor(2.5,device=self.device)
        # self._commands[env_ids, 0] += radius * torch.cos(angle)
        # self._commands[env_ids, 1] += radius * torch.sin(angle)
        # self._commands[env_ids, 2] = 0.6
        # self._commands[env_ids, 3] = 5
        
        self.goal_visualizer.visualize(translations=self._commands[:,:3])
        # command_values = torch.tensor([0.5, -0.3, self._robot.data.default_root_state[env_ids, 2], 5.0] , device=self.device) # command to position X Y Z and time remaining
        # self._commands[env_ids] = command_values.unsqueeze(0).expand(len(env_ids), -1)
        # print(f"command_Assign{self._commands}")
        ################################################################################################################
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
        