# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlDppoAlgorithmCfg, RslRlSymmetryCfg ,RslRlPpoActorCriticRecurrentCfg,RslRlPpoQuantileCfg ,RslRlEncoderCfg

@configclass
class AnymalCFlatDPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 50
    experiment_name = "GMT_locomotion_risk_fix"
    empirical_normalization = False
    policy = RslRlPpoQuantileCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
        rnn_hidden_dim=128,
        rnn_type="lstm",
        rnn_num_layers=1,
        measure_kwargs={"beta":1.5},
        quantile_count = 400,
        encoder_obs = False,
        is_beta = True,
    )

    # encoder = RslRlEncoderCfg(
    #     type = "gru",
    #     hidden_dims=[256, 128],
    #     output_dim=8,
    #     obs_indices = 36,
    #     gru_hidden_size = 256,
    #     gru_num_layers = 2,
    #     privilaged_obs_indices = 0,
    # )

    algorithm = RslRlDppoAlgorithmCfg(
        class_name="DPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        distributional_loss_type="energy",
    )

@configclass
class AnymalCRoughDPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "Anymal_locomotion_risk_fix"
    empirical_normalization = False
    policy = RslRlPpoQuantileCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        # actor_hidden_dims=[128, 128, 128],
        # critic_hidden_dims=[128, 128, 128],
        activation="elu",
        rnn_hidden_dim=128,
        rnn_type="lstm",
        rnn_num_layers=1,
        measure_kwargs={"beta":1.5},
        quantile_count = 400,
        encoder_obs = True,
        is_beta = True,
    )

    encoder = RslRlEncoderCfg(
        type = "mlp",
        hidden_dims=[256, 128],
        output_dim=64,
        obs_indices = 49,
        # gru_hidden_size = 256,
        # gru_num_layers = 2,
        privilaged_obs_indices = 1,
    )

    algorithm = RslRlDppoAlgorithmCfg(
        class_name="DPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        distributional_loss_type="energy",
    )
  

@configclass
class AnymalCFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 50
    experiment_name = "anymal_c_flat_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class AnymalCRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "anymal_c_rough_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


