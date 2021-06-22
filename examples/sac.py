import gym
import pybulletgym

ENVS = {'HalfCheetahPyBulletEnv-v0', 'HopperPyBulletEnv-v0',
        'AntPyBulletEnv-v0', 'Walker2DPyBulletEnv-v0'}

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import torch
import os


def experiment(variant, args):
    env = gym.make(args.env)
    expl_env = NormalizedBoxEnv(env)
    eval_env = NormalizedBoxEnv(env)
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )

    if args.eval:
        policy.load_state_dict(torch.load(os.path.join(args.model_path, 'policy_{}.pth'.format(args.eval_epoch))))
        qf1.load_state_dict(torch.load(os.path.join(args.model_path, 'qf1_{}.pth'.format(args.eval_epoch))))
        qf2.load_state_dict(torch.load(os.path.join(args.model_path, 'qf2_{}.pth'.format(args.eval_epoch))))

    # For sampling, we shouod use stochastic policy instead
    eval_policy = MakeDeterministic(policy) if not args.eval else policy
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)

    if args.eval:
        print('Genearting model rollouts...')
        save_path = os.path.join(args.model_path, "traj_sample_{}.pkl".format(args.eval_epoch))
        algorithm.eval(save_path)
    else:
        algorithm.train()

if __name__ == "__main__":
    # noinspection PyTypeChecker

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='debug')
    parser.add_argument('--env', choices=ENVS, default='HopperPyBulletEnv-v0')
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--eval_epoch', type=int, default=0)
    parser.add_argument('--num_eval_steps', type=int, default=5000)
    parser.add_argument('--model_path', type=str, default='/mnt/efs/Projects/rlkit-pybullet/data/offline-data/offline_data_2021_06_21_21_33_07_0000--s-0')
    args = parser.parse_args()

    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        env_name = args.env,
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=args.num_eval_steps,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger(args.exp_name, variant=variant, snapshot_gap=50, snapshot_mode='gap')
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant, args)
