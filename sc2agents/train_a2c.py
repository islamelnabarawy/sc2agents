#!/usr/bin/env python
import logging
import os
import sys

import gym
# noinspection PyUnresolvedReferences
import sc2gym.envs
from absl import flags
from baselines import bench
from baselines import logger
from baselines.a2c.a2c import learn
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

FLAGS = flags.FLAGS


def train(env_id, num_timesteps, seed, policy, lrschedule, num_cpu):
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            gym.logger.setLevel(logging.WARN)
            return env

        return _thunk

    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule)
    env.close()


def main():
    FLAGS(sys.argv)

    logger.configure()

    train('SC2MoveToBeacon-v0', num_timesteps=int(10e6), seed=0,
          policy='cnn', lrschedule='constant', num_cpu=4)


if __name__ == '__main__':
    main()
