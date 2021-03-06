import tensorflow as tf
from replay_ppo2 import ppo2
from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from procgen_replay.procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from baselines import logger
from mpi4py import MPI
import argparse
import numpy as np

def train_fn(env_name, num_envs, distribution_mode, num_levels, start_level, timesteps_per_proc, level_sampler_strategy, score_transform, model_name, is_test_worker=False, save_dir='./', comm=None):
    learning_rate = 5e-4
    ent_coef = .01
    gamma = .999
    lam = .95
    nsteps = 256
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    use_vf_clipping = True

    mpi_rank_weight = 0 if is_test_worker else 1
    num_levels = 0 if is_test_worker else num_levels

    log_dir = save_dir + 'logs/' + model_name

    if log_dir is not None:
        log_comm = comm.Split(1 if is_test_worker else 0, 0)
        format_strs = ['csv', 'stdout','tensorboard'] if log_comm.Get_rank() == 0 else []
        logger.configure(comm=log_comm, dir=log_dir, format_strs=format_strs)

    logger.info("creating environment")
    eval_env = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=500, start_level=0, distribution_mode=distribution_mode)
    eval_env = VecExtractDictObs(eval_env, "rgb")
    eval_env = VecMonitor(
        venv=eval_env, filename=None, keep_buf=100,
    )
    eval_env = VecNormalize(venv=eval_env, ob=False, ret=True)

    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32])

    logger.info("training")
    model = ppo2.learn(
        network=conv_fn,
        total_timesteps=timesteps_per_proc,
        num_levels=num_levels,
        eval_env=eval_env,
        save_interval=0,
        nsteps=nsteps,
        nminibatches=nminibatches,
        lam=lam,
        gamma=gamma,
        noptepochs=ppo_epochs,
        log_interval=1,
        ent_coef=ent_coef,
        mpi_rank_weight=mpi_rank_weight,
        clip_vf=use_vf_clipping,
        comm=comm,
        lr=learning_rate,
        cliprange=clip_range,
        update_fn=None,
        init_fn=None,
        vf_coef=0.5,
        max_grad_norm=0.5,
        level_sampler_strategy=level_sampler_strategy,
        score_transform=score_transform
    )
    model.save(save_dir + 'models/' + model_name)

def main():
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=50)
    parser.add_argument('--start_level', type=int, default=500)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--timesteps_per_proc', type=int, default=1_000_000)
    parser.add_argument('--level_sampler_strategy', type=str, default='value_l1')
    parser.add_argument('--score_transform', type=str, default='rank')
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--save_dir', type=str, default='gdrive/MyDrive/182 Project/')

    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False
    test_worker_interval = args.test_worker_interval

    if test_worker_interval > 0:
        is_test_worker = rank % test_worker_interval == (test_worker_interval - 1)

    train_fn(args.env_name,
        args.num_envs,
        args.distribution_mode,
        args.num_levels,
        args.start_level,
        args.timesteps_per_proc,
        args.level_sampler_strategy,
        args.score_transform,
        args.model_name,
        is_test_worker=is_test_worker,
        save_dir=args.save_dir, 
        comm=comm)

if __name__ == '__main__':
    main()
