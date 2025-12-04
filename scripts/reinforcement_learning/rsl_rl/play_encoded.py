# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.modules import obs_encoder as encoders

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import GMT_robotParkour.tasks  # noqa: F401


def main():
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # fix random seed and setup camera view
    # env_cfg.viewer.eye = (-1.0, -10.0, 6.0)
    # env_cfg.viewer.lookat = (-1.0, 0.0, 0.0)
    env_cfg.seed = agent_cfg.seed
    # env_cfg.viewer.eye = (0.0, -5.0, 1.0)
    # env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    # env_cfg.viewer.origin_type = "asset_body"
    # env_cfg.viewer.env_index = 0
    # env_cfg.viewer.asset_name = "robot"
    # env_cfg.viewer.body_name = "base"
    # env_cfg.viewer.eye = (-1.0, 0.5, 15.0)
    # env_cfg.viewer.lookat = (-1.0, 0.5, 0.0)
    # env_cfg.viewer.origin_type = "env"
    # env_cfg.viewer.env_index = 0
    # env_cfg.viewer.asset_name = "robot"
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # setup videe log path
    # setup = "DPPO0.0toDPPO1.5_Obstacle"
    # setup = "DPPO1.5toDPPO1.5_Obstacle"
    
    # setup beta for test
    beta = 1.5
    setup = f"Risk_{beta}"

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", setup),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    
    

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # policy2 = torch.jit.load(os.path.join(export_model_dir, "policy.pt")).to(agent_cfg.device)
    # policy2.eval()
    dt = env.unwrapped.step_dt

    # reset environment

    # setup encoder model for play
    encoder_cfg = agent_cfg.to_dict()["encoder"]
    index = encoder_cfg.get("obs_indices",36)
    obs, infos = env.get_observations()
    timestep = 0
    obs_indices = obs[:,index:]
    input_dim = obs_indices.shape[1]
    output_dim = encoder_cfg.get("output_dim", 8)
    device = "cuda"
    # print(f'obs shape:{obs.shape}')
    # print(f'obs_indices shape:{obs_indices.shape}')
    encoder_params = {
                "input_dim": input_dim,
                "output_dim": output_dim,
                "hidden_dims": encoder_cfg.get("hidden_dims", [256, 128])
    }
    if encoder_cfg.get("type","mlp") == "mlp":
        obs_encoder = encoders.ObsEncoder(**encoder_params).to(device)
        print(f"MLP Encoder Structure: {obs_encoder}")    
    elif encoder_cfg.get("type","mlp") == "gru":
        encoder_params.update({
            "gru_hidden_size": encoder_cfg.get("gru_hidden_size", 256),
            "gru_num_layers": encoder_cfg.get("gru_num_layers", 2)
        })
        obs_encoder = encoders.GRUEncoder(**encoder_params).to(device)
        print(f"GRU Encoder Structure: {obs_encoder}")
    elif encoder_cfg.get("type","mlp") == "conv":
        encoder_params.update({
            "conv_channels": encoder_cfg.get("conv_channels", [32, 64, 128]),
            "conv_kernel_sizes": encoder_cfg.get("conv_kernel_sizes", [3, 3, 3]),
            "conv_strides": encoder_cfg.get("conv_strides", [1, 1, 1])
        })
        obs_encoder = encoders.ConvEncoder(**encoder_params).to(device)
        print(f"Conv Encoder Structure: {obs_encoder}")

    elif encoder_cfg.get("type","mlp") == "convgru":
        encoder_params.update({
            "conv_channels": encoder_cfg.get("conv_channels", [32, 64]),
            "conv_kernel_sizes": encoder_cfg.get("conv_kernel_sizes", [3, 3]),
            "pool_sizes": encoder_cfg.get("pool_sizes", [2, 2, 2]),
            "gru_hidden_size": encoder_cfg.get("gru_hidden_size", 256),
            "gru_num_layers": encoder_cfg.get("gru_num_layers", 1)
        })
        obs_encoder = encoders.ConvGRUEncoder(**encoder_params).to(device)
        print(f"ConvGRU Encoder Structure: {obs_encoder}")

    obs_encoder.load_state_dict(torch.load(resume_path)["encoder_state_dict"])
    obs_encoder.eval()

    # export encoder to .jit model
    dummy_obs = torch.zeros(1, obs_indices.shape[1], device=agent_cfg.device)
    traced_encoder = torch.jit.trace(obs_encoder, dummy_obs)
    traced_encoder.save(os.path.join(export_model_dir, "encoder.pt"))
    # export_policy_as_jit(obs_encoder,None, path=export_model_dir, filename="encoder.pt")
    # simulate environment
    # Prepare logging for Episode_Termination/base_contact
    base_contact_value = []
    base_contact_log_path = os.path.join(log_dir, "videos", setup, "base_contact_log.txt")
    os.makedirs(os.path.dirname(base_contact_log_path), exist_ok=True)
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # print(f"obs shape:{obs}")
            # agent stepping
            selected_obs = obs[:,index:]
            # print(f'Selected obs shape:{obs}')
            encoded_obs = obs_encoder(selected_obs)
            # print(f'Encoded obs:{encoded_obs}')
            modified_obs = obs.clone()
            modified_obs = modified_obs[:,:index]
            # print(f'modified_obs shape:{modified_obs}')
            modified_obs[:,-1] = beta
            modified_obs = torch.cat([modified_obs, encoded_obs], dim=1)
            # print(f'modified_obs:{modified_obs}')

            # actions = policy(modified_obs) # for encoder
            actions = policy_nn.act_inference(modified_obs) # for RNN policy (Need to reset after end episode or got terminate)
            # actions = policy(obs) #for no encoder

            # env stepping
            # print(f'action:{actions}')
            obs, rewards, dones, infos = env.step(actions)
           
            #reset policy RNN
            policy_nn.reset(dones)
            timestep += 1
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                # with open(base_contact_log_path, "a") as f:
                #     f.write(str(max(base_contact_value)) + "\n")
                break
        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
