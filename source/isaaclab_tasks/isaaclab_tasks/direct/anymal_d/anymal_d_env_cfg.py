# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, ImuCfg , CameraCfg,RayCasterCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg , TerrainGeneratorCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
# from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab.terrains.config.box import BOX_TERRAINS_CFG , MULTI_TERRAINS_BOX_CFG, MULTI_TERRAINS_PIT_CFG ,MULTI_TERRAINS_PLANE_CFG , PIT_TERRAINS_CFG


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )


@configclass
class AnymalDFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 12
    observation_space = 48
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=True,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # reward scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -5.
    robot_height_reward_scale = 0


@configclass
class AnymalDRoughEnvCfg(AnymalDFlatEnvCfg):
    # env
    observation_space = 235

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    imu_sensor = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        update_period = 0.1,
    )

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0


@configclass
class AnymalDClimbEnvCfg(AnymalDFlatEnvCfg):
    # env
    observation_space = 235

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=BOX_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        # pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
            intrinsic_matrix= [525.0, 0.0, 319.5,
                                0.0, 525.0, 239.5,
                                0.0, 0.0, 1.0],
            width=640,
            height=480,
              ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0
    robot_height_reward_scale = -1.
    

@configclass
class AnymalDFlatEnvPosCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 6.0
    decimation = 4
    action_scale = 0.5
    action_space = 12
    observation_space = 235
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=MULTI_TERRAINS_PLANE_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    #     debug_vis=True,
    # )


    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.0, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    imu_sensor = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        update_period = 0.1,
    )

    camera_cfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/base/CameraSensor",
        update_period=0,
        height=480,
        width=640,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "normals",
            "semantic_segmentation",
            "instance_segmentation_fast",
            "instance_id_segmentation_fast",
        ],
        colorize_semantic_segmentation=True,
        colorize_instance_id_segmentation=True,
        colorize_instance_segmentation=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.2, 0.0, 0.5),rot=(0.354,-0.612,0.612,-0.354)) 
    )

    raycamera_cfg = RayCasterCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/CameraSensor",
        mesh_prim_paths=["/World/ground"],
        update_period=0.1,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        data_types=["distance_to_image_plane", "normals", "distance_to_camera"],
        debug_vis=True,
        depth_clipping_behavior= "zero",
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=48,
            width=64,
        ),
        # pattern_cfg=patterns.PinholeCameraPatternCfg(
        #     focal_length=1.93,
        #     horizontal_aperture=4.85,
        #     height=720,
        #     width=1280,
        # ),
    )
    # # Create camera
    # camera = Camera(cfg=camera_cfg)   

    # return camera

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # reward scales
    # lin_vel_reward_scale = 1.0
    # yaw_rate_reward_scale = 0.5
    # z_vel_reward_scale = -2.0
    # ang_vel_reward_scale = -0.05
    # joint_torque_reward_scale = -2.5e-5
    # joint_accel_reward_scale = -2.5e-7
    # action_rate_reward_scale = -0.01
    # feet_air_time_reward_scale = 0.5
    # undesired_contact_reward_scale = -1.0
    # flat_orientation_reward_scale = -5.
    # robot_height_reward_scale = 0
    # task_reward_scale = 1.0
    # bias_reward_scale = 1.0
    # stall_reward_scale = 1.0
    # feet_accel_reward_scale = -0.000003
    # heading_reward_scale = 0.1


    # joint_torque_reward_scale = -0.00001
    # # joint_accel_reward_scale = -2.5e-7
    # joint_velo_reward_scale =  -0.001
    # action_rate_reward_scale = -0.01
    # action_rate2_reward_scale = -0.01
    # undesired_contact_reward_scale = -1.0
    # flat_orientation_reward_scale = 0.0
    # task_reward_scale = 10.0
    # bias_reward_scale = 3.0
    # stall_reward_scale = -1.0
    # feet_accel_reward_scale = -0.0002
    # heading_reward_scale = 1.0
    # heading_command_reward_scale = 0.0
    # terminate_reward_scale = -200
    # feet_force_reward_scale = -0.002
    # base_accel_reward_scale = -0.0001
    # stumble_reward_scale = -1
    # stand_target_reward_scale = -0.5
    # velo_limit_reward_scale = -1
    # joint_limit_reward_scale = -0.2

    task_reward_scale = 10.0
    heading_command_reward_scale = 5.0
    joint_velo_reward_scale =  -0.001
    joint_torque_reward_scale = -0.00001
    velo_limit_reward_scale = -1
    joint_limit_reward_scale = -0.2
    base_accel_reward_scale = -0.001
    feet_accel_reward_scale = -0.002
    action_rate_reward_scale = -0.01
    feet_force_reward_scale = -0.00001
    stall_reward_scale = -1.0
    bias_reward_scale = 1.0
    stand_target_reward_scale = 0.5
    undesired_contact_reward_scale = -1.0
    stumble_reward_scale = -1.0
    terminate_reward_scale = -200
    heading_reward_scale = 0.0
    action_rate2_reward_scale = 0.0
    flat_orientation_reward_scale = 0.0
    

    # joint_torque_reward_scale = -1.0e-05
    # # joint_accel_reward_scale = -2.5e-07
    # joint_velo_reward_scale = -0.001
    # action_rate_reward_scale = -0.01S
    # undesired_contact_reward_scale = -1.0
    # task_reward_scale = 10.0
    # bias_reward_scale = 3.0
    # stall_reward_scale = -1.0
    # feet_accel_reward_scale = -0.0002
    # heading_reward_scale = 1.0
    # terminate_reward_scale = -200
    # feet_force_reward_scale = -1.0e-05
    # # base_accel_reward_scale = -0.0001
    # stumble_reward_scale = -1

    # joint_torque_reward_scale: -1.0e-05
    # joint_velo_reward_scale: -0.001
    # action_rate_reward_scale: -0.01
    # undesired_contact_reward_scale: -1.0
    # task_reward_scale: 10.0
    # bias_reward_scale: 3.0
    # stall_reward_scale: -1.0
    # feet_accel_reward_scale: -0.0002
    # heading_reward_scale: 1.0
    # heading_command_reward_scale: 5.0
    # terminate_reward_scale: -200
    # feet_force_reward_scale: -1.0e-05
    # base_accel_reward_scale: -0.001
    # stumble_reward_scale: -1
    # stand_target_reward_scale: -0.5
    # velo_limit_reward_scale: -1
    # joint_limit_reward_scale: -0.2

@configclass
class AnymalDClimbUpEnvPosCfg(AnymalDFlatEnvPosCfg):
    # env
    observation_space = 235
    episode_length_s = 10.0
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     terrain_generator=BOX_TERRAINS_CFG,
    #     max_init_terrain_level=1,
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     visual_material=sim_utils.MdlFileCfg(
    #         mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
    #         project_uvw=True,
    #     ),
    #     debug_vis=False,
    # )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=MULTI_TERRAINS_PIT_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.0, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # reward scales (override from flat config)
    # lin_vel_reward_scale = 1.0
    # yaw_rate_reward_scale = 0.5
    # z_vel_reward_scale = -2.0
    # ang_vel_reward_scale = -0.05
    # joint_torque_reward_scale = -2.5e-5
    # joint_accel_reward_scale = -2.5e-7
    # action_rate_reward_scale = -0.01
    # feet_air_time_reward_scale = 0.5
    # undesired_contact_reward_scale = 0
    # flat_orientation_reward_scale = 0
    # robot_height_reward_scale = 0
    # task_reward_scale = 1.0
    # bias_reward_scale = 1.0
    # stall_reward_scale = 1.0
    # feet_accel_reward_scale = -0.0000003
    # heading_reward_scale = 0.1

    task_reward_scale = 10.0
    heading_command_reward_scale = 5.0
    joint_velo_reward_scale =  -0.001
    joint_torque_reward_scale = -0.00001
    velo_limit_reward_scale = -1
    joint_limit_reward_scale = -0.2
    base_accel_reward_scale = -0.001
    feet_accel_reward_scale = -0.002
    action_rate_reward_scale = -0.01
    feet_force_reward_scale = -0.00001
    stall_reward_scale = -1.0
    bias_reward_scale = 1.0
    stand_target_reward_scale = 0.5
    undesired_contact_reward_scale = -0.1
    stumble_reward_scale = -1.5
    terminate_reward_scale = -200
    heading_reward_scale = 0.0
    action_rate2_reward_scale = 0.0
    flat_orientation_reward_scale = 0.0

@configclass
class AnymalDClimbDownEnvPosCfg(AnymalDFlatEnvPosCfg):
    # env
    observation_space = 235

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        # terrain_generator=BOX_TERRAINS_CFG,
        terrain_generator=MULTI_TERRAINS_BOX_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.0, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # reward scales (override from flat config)
    # lin_vel_reward_scale = 1.0
    # yaw_rate_reward_scale = 0.5
    # z_vel_reward_scale = -2.0
    # ang_vel_reward_scale = -0.05
    # joint_torque_reward_scale = -2.5e-5
    # joint_accel_reward_scale = -2.5e-7
    # action_rate_reward_scale = -0.01
    # feet_air_time_reward_scale = 0.5
    # undesired_contact_reward_scale = 0 #-1.0
    # flat_orientation_reward_scale = 0
    # robot_height_reward_scale = 0
    # task_reward_scale = 1.0
    # bias_reward_scale = 1.0
    # stall_reward_scale = 1.0
    # feet_accel_reward_scale = -0.000003
    # heading_reward_scale = 0.1
    # joint_torque_reward_scale = -0.00001
    # # joint_accel_reward_scale = -2.5e-7
    # joint_velo_reward_scale =  -0.001
    # action_rate_reward_scale = -0.01
    # undesired_contact_reward_scale = -1.0
    # # flat_orientation_reward_scale = -5.
    # task_reward_scale = 10.0
    # bias_reward_scale = 1.0
    # stall_reward_scale = -1.0
    # feet_accel_reward_scale = -0.002
    # heading_reward_scale = 1.0
    # heading_command_reward_scale = 5.0
    # terminate_reward_scale = -200
    # feet_force_reward_scale = -0.00001
    # base_accel_reward_scale = -0.001
    # stumble_reward_scale = -1
    # stand_target_reward_scale = -0.5
    # velo_limit_reward_scale = -1
    # joint_limit_reward_scale = -0.
    
    # joint_torque_reward_scale = -0.00001
    # # joint_accel_reward_scale = -2.5e-7
    # joint_velo_reward_scale =  -0.001
    # action_rate_reward_scale = -0.01
    # action_rate2_reward_scale = -0.01
    # undesired_contact_reward_scale = -1.0
    # # flat_orientation_reward_scale = -5.
    # task_reward_scale = 10.0
    # bias_reward_scale = 2.0
    # stall_reward_scale = -1.0
    # feet_accel_reward_scale = -0.00035 #-0.002
    # heading_reward_scale = 1.0
    # heading_command_reward_scale = 5.0
    # terminate_reward_scale = -200
    # feet_force_reward_scale = -0.002
    # base_accel_reward_scale = -0.001
    # stumble_reward_scale = -1
    # stand_target_reward_scale = -0.5
    # velo_limit_reward_scale = -1
    # joint_limit_reward_scale = -0.2

    task_reward_scale = 10.0
    heading_command_reward_scale = 5.0
    joint_velo_reward_scale =  -0.001
    joint_torque_reward_scale = -0.00001
    velo_limit_reward_scale = -1
    joint_limit_reward_scale = -0.2
    base_accel_reward_scale = -0.001
    feet_accel_reward_scale = -0.002
    action_rate_reward_scale = -0.01
    feet_force_reward_scale = -0.00001
    stall_reward_scale = -1.0
    bias_reward_scale = 1.0
    stand_target_reward_scale = 0.5
    undesired_contact_reward_scale = -0.1
    stumble_reward_scale = -1.0
    terminate_reward_scale = -200
    heading_reward_scale = 0.0
    action_rate2_reward_scale = 0.0
    flat_orientation_reward_scale = 0.0
