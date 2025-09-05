# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import isaaclab.utils.math as math_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor , RayCaster , Imu , Camera, RayCasterCamera
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.markers import VisualizationMarkers

from .anymal_d_env_cfg import AnymalDClimbUpEnvPosCfg , AnymalDClimbDownEnvPosCfg, AnymalDFlatEnvPosCfg

class AnymalDEnvPos(DirectRLEnv):
    cfg: AnymalDFlatEnvPosCfg | AnymalDClimbUpEnvPosCfg | AnymalDClimbDownEnvPosCfg

    def __init__(self, cfg: AnymalDFlatEnvPosCfg | AnymalDClimbUpEnvPosCfg | AnymalDClimbDownEnvPosCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._previous_actions2 = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self.reward_buffer = torch.zeros(self.num_envs, device=self.device)
        # X Y Z position and Time remaining
        self._commands = torch.zeros(self.num_envs, 5, device=self.device)
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
                "dof_torques_l2",
                "dof_vel_l2",
                "action_rate_l2",
                "action_rate2_l2",
                "undesired_contacts",
                "task_reward",
                "bias_reward",
                "stall_reward",
                "feet_accel",
                "heading_command",
                "termination",
                "feet_contact_force",
                "base_accel",
                "stumble",
                "stand_target",
                "velocity_limit",
                "joint_limit",
                "heading",
                "flat_orientation",
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
        # we add a height scanner for perceptive locomotion
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        self._imu_sensor = Imu(self.cfg.imu_sensor)
        self.scene.sensors["imu_sensor"] = self._imu_sensor

        self._camera = Camera(self.cfg.camera_cfg)
        self.scene.sensors["camera"] = self._camera

        self._raycamera = RayCasterCamera(self.cfg.raycamera_cfg)
        self.scene.sensors["raycamera"] = self._raycamera
        # self._camera.set_world_poses_from_view(torch.tensor([[2.5, 2.5, 2.5], [-2.5, -2.5, 2.5]], device=self.device)
        #                                        , torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=self.device))

        self.i = 0
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        # get target from terrain sampling
        self.valid_targets: torch.Tensor = self._terrain.flat_patches["target"]

        
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        ################################################################################################################
        self._previous_actions2 = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()
        height_data = (
            self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
        ).clip(-1.0, 1.0)

        # scan_data = self._raycamera.data.output["distance_to_image_plane"]
        # self._raycamera.data
        # prepare command for obs
        commandobs = torch.clone(self._commands)
        commandobs[:, :3] = self._commands[:, :3]-self._robot.data.root_pos_w[:, :3]
        commandobs[:, 3] = torch.abs(self._robot.data.heading_w - ((self._commands[:,3] + torch.pi) % (2 * torch.pi) - torch.pi))
        commandobs[:,4] = (self._commands[:,4] - self.episode_length_buf)/50
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    # self._actions,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    commandobs,
                    height_data,
                    
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
        # print(self._camera.data.output["distance_to_image_plane"].shape)
        depth_image = self._camera.data.output["distance_to_image_plane"][1, :, :, 0].cpu().numpy()  # shape: (480, 640)
        # depth_image = self._camera.data.output["rgb"][1, :, :, 0].cpu().numpy()
        if self.i == 0:
            plt.ion()  # interactive mode
            self.fig, self.ax = plt.subplots()
            self.im = self.ax.imshow(depth_image)
            # self.im = self.ax.imshow(depth_image,cmap='plasma', vmin=0, vmax=3)  # fix vmin/vmax for consistent colors
            plt.colorbar(self.im, ax=self.ax, label='Depth (m)')
            plt.title("Real-Time Depth Image")
            self.i = 1
        else:
            self.im.set_data(depth_image)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        # joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        joint_velo = torch.sum(torch.square(self._robot.data.joint_vel), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        action_rate2 = torch.sum(torch.square(self._actions - 2*self._previous_actions + self._previous_actions2), dim=1)
        # base acceleration
        base_accel = (torch.square(torch.norm(self._robot.data.body_lin_acc_w[:,0],dim=1)) + 
                    (0.02 * torch.square(torch.norm(self._robot.data.body_ang_acc_w[:,0],dim=1)))) #####
        # joint velocity limit
        joint_velo_limit = torch.sum(torch.maximum(torch.abs(self._robot.data.joint_vel)-self._robot.data.joint_vel_limits,torch.zeros_like(self._robot.data.joint_vel)),dim=1)
        # joint torque limit
        joint_torque_limit = torch.sum(torch.maximum(torch.abs(self._robot.data.applied_torque)-self._robot.data.joint_effort_limits,torch.zeros_like(self._robot.data.joint_vel)),dim=1)
        # feet accelarate
        feet_accel = torch.sum(torch.norm(self._robot.data.body_lin_acc_w[:,13:17],dim=1),dim=1)
        # feet_accel = torch.sum(torch.square(torch.norm(self._robot.data.body_lin_acc_w[:,13:17],dim=1)),dim=1)
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
            is_contact3 = torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0
            contacts = torch.sum(is_contact, dim=1)
            contacts2 = torch.sum(is_contact2, dim=1)
            contacts3 = torch.sum(is_contact3, dim=1)
            contacts += contacts2
            contacts += contacts3 
        # foot force penalty
        foot_force_norms = torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids],dim=1)
        feet_contact_force = torch.sum(torch.square(torch.maximum(foot_force_norms-700,torch.zeros_like(foot_force_norms))),dim=1)
        stumble_mask = torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids,:2],dim=-1) > torch.abs(self._contact_sensor.data.net_forces_w[:, self._feet_ids,2]) * 2
        stumble_penalty = torch.where(stumble_mask, torch.full_like(stumble_mask, 1.0), torch.zeros_like(stumble_mask))
        stumble_penalty = torch.sum(stumble_penalty,dim=1)

        # Compute time condition per environment
        time_condition = (self._commands[:,4] - self.episode_length_buf)/50 < 1
        # Compute task reward per environment
        reward_task = torch.where(
            time_condition,
            1-(0.5*torch.norm(self._robot.data.root_pos_w[:, :2] - self._commands[:, :2],dim=1)),
            torch.zeros(self.num_envs, device=self.device)
        )
        non_zero_mask = reward_task != 0
        self.reward_buffer[non_zero_mask] = reward_task[non_zero_mask]
        # heading tracking command
        command_map = (self._commands[:,3] + torch.pi) % (2 * torch.pi) - torch.pi
        reward_heading = torch.where(
            time_condition,
            1-(0.5*torch.abs(self._robot.data.heading_w-command_map)),
            torch.zeros(self.num_envs, device=self.device)
        )
        self.reward_task_reached |= reward_task >= 0.65
        # bias reward
        if (~(self.reward_task_reached.all() and torch.mean(self.reward_buffer)> 0.5)):
            reward_bias = (
            torch.sum(self._robot.data.root_lin_vel_b[:, :3] * (self._commands[:, :3] - self._robot.data.root_pos_w[:, :3]), dim=1) /
            (torch.norm(self._robot.data.root_lin_vel_b[:, :3], dim=1) *
             torch.norm(self._commands[:, :3] - self._robot.data.root_pos_w[:, :3], dim=1))
            )
        else:
            reward_bias = torch.zeros(self.num_envs,device=self.device)
        # Stand at target
        reach_condition = torch.norm(self._robot.data.root_pos_w[:, :2] - self._commands[:, :2],dim=1) < 0.25
        # r_heading = torch.cos(yaw - desired_heading) #take cos for reward 
        reach_condition &= torch.abs(self._robot.data.heading_w-command_map) < 0.5
        # reach_condition = torch.abs(self._robot.data.heading_w-command_map) < 0.5
        stand_target = torch.where(
            reach_condition,
            torch.exp(-torch.sum((self._robot.data.joint_pos-self._robot.data.default_joint_pos) ** 2, dim=1)),
            torch.zeros(self.num_envs, device=self.device)
        )
        # Compute per-env values
        velocity = torch.norm(self._robot.data.root_lin_vel_b[:, :3], dim=1)
        pos_error = torch.norm(self._commands[:, :3] - self._robot.data.root_pos_w[:, :3], dim=1)
        # Condition: low velocity AND high position error, per robot
        stall_mask = (velocity < 0.2) & (pos_error > 0.5)
        # stall_mask = (velocity < 0.2)
        # Assign -1 reward where the condition is met
        reward_stall = torch.where(stall_mask, torch.full_like(velocity, 1.0), torch.zeros_like(velocity))

        # rewards = {
        #     "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale  ,
        #     "dof_vel_l2" : joint_velo * self.cfg.joint_velo_reward_scale * self.step_dt,
        #     "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
        #     "action_rate2_l2" : action_rate2 * self.cfg.action_rate2_reward_scale * self.step_dt,
        #     "heading_command" : reward_heading  * self.cfg.heading_command_reward_scale * self.step_dt,
        #     "base_accel" : base_accel * self.cfg.base_accel_reward_scale * self.step_dt,
        #     "feet_accel" : feet_accel * self.cfg.feet_accel_reward_scale * self.step_dt,
        #     "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt, ####
        #     "task_reward" : reward_task*self.cfg.task_reward_scale * self.step_dt,
        #     "bias_reward" : reward_bias*self.cfg.bias_reward_scale * self.step_dt,
        #     "stall_reward" : reward_stall*self.cfg.stall_reward_scale * self.step_dt, ####
        #     "termination" : self.cfg.terminate_reward_scale * self.reset_terminated.float(),
        #     "feet_contact_force" : self.cfg.feet_force_reward_scale * feet_contact_force * self.step_dt,
        #     "stumble" : self.cfg.stumble_reward_scale * stumble_penalty * self.step_dt,
        #     "stand_target" : self.cfg.stand_target_reward_scale * stand_target * self.step_dt,
        #     "velocity_limit" : self.cfg.velo_limit_reward_scale * joint_velo_limit * self.step_dt,
        #     "joint_limit" : self.cfg.joint_limit_reward_scale * joint_torque_limit * self.step_dt,
        #     }
        rewards = {
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale ,
            "dof_vel_l2" : joint_velo * self.cfg.joint_velo_reward_scale ,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale ,
            "action_rate2_l2" : action_rate2 * self.cfg.action_rate2_reward_scale ,
            "heading_command" : reward_heading  * self.cfg.heading_command_reward_scale ,
            "base_accel" : base_accel * self.cfg.base_accel_reward_scale,
            "feet_accel" : feet_accel * self.cfg.feet_accel_reward_scale ,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale  , ####
            "task_reward" : reward_task*self.cfg.task_reward_scale  ,
            "bias_reward" : reward_bias*self.cfg.bias_reward_scale  ,
            "stall_reward" : reward_stall*self.cfg.stall_reward_scale  , ####
            "termination" : self.cfg.terminate_reward_scale * self.reset_terminated.float(),
            "feet_contact_force" : self.cfg.feet_force_reward_scale * feet_contact_force  ,
            "stumble" : self.cfg.stumble_reward_scale * stumble_penalty  ,
            "stand_target" : self.cfg.stand_target_reward_scale * stand_target  ,
            "velocity_limit" : self.cfg.velo_limit_reward_scale * joint_velo_limit  ,
            "joint_limit" : self.cfg.joint_limit_reward_scale * joint_torque_limit  ,
            }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward
        ################################################################################################################
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        # died = (torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1))
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._feet_ids], dim=-1), dim=1)[0] > 1500.0, dim=1)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
            # self.episode_length_buf[:] = torch.zeros_like(self.episode_length_buf)
            self.episode_temp = torch.clone(self.episode_length_buf)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        ################################################################################################################
        # Sample new commands
        if  isinstance(self.cfg, AnymalDFlatEnvPosCfg):
            ids = torch.randint(0, self.valid_targets.shape[2], size=(len(env_ids),), device=self.device)
            # angle = torch.rand(1,device=self.device) * 2 * torch.pi
            # radius = 1 + 4 * torch.rand(1,device=self.device)   
            self._commands[env_ids,:3] =  self.valid_targets[self._terrain.terrain_levels[env_ids], self._terrain.terrain_types[env_ids],ids]
            angle = torch.atan2(self._commands[env_ids, 1] - self._terrain.env_origins[env_ids,1], self._commands[env_ids, 0] - self._terrain.env_origins[env_ids,0])
            # self._commands[env_ids,0] = self._terrain.env_origins[env_ids,0] + radius * torch.cos(angle)
            # self._commands[env_ids,1] = self._terrain.env_origins[env_ids,1] + radius * torch.sin(angle)
            self._commands[env_ids,2] += 0.6
            default_root_state = self._robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        if  isinstance(self.cfg, AnymalDClimbUpEnvPosCfg):
            angle = torch.rand(1,device=self.device) * 2 * torch.pi
            radius = torch.tensor(4.5,device=self.device)
            self._commands[env_ids,0] = self._terrain.env_origins[env_ids,0] + radius * torch.cos(angle)
            self._commands[env_ids,1] = self._terrain.env_origins[env_ids,1] + radius * torch.sin(angle)
            self._commands[env_ids,2] = 0.6
            default_root_state = self._robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        if  isinstance(self.cfg, AnymalDClimbDownEnvPosCfg):
            angle = torch.rand(1,device=self.device) * 2 * torch.pi
            radius = torch.tensor(4.5,device=self.device)
            self._commands[env_ids,0] = self._terrain.env_origins[env_ids,0] + radius * torch.cos(angle)
            self._commands[env_ids,1] = self._terrain.env_origins[env_ids,1] + radius * torch.sin(angle)
            self._commands[env_ids,2] = 0.6
            default_root_state = self._robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # self._commands[env_ids,3] = torch.rand(1,device=self.device) * 2 * torch.pi
        self._commands[env_ids,3] = angle
        self._commands[env_ids,4] = torch.tensor(self.max_episode_length,device=self.device,dtype=torch.float32)
        zeros = torch.zeros_like(self._commands[:,3],device=self.device)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, self._commands[:,3])
        self.goal_visualizer.visualize(translations=self._commands[:,:3],orientations=arrow_quat)
        # print(f"command_Assign{self._commands}")
        # print(f"command_Assign{arrow_quat}")
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
        