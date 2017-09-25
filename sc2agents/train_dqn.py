import sys

import gflags as flags
from baselines import deepq

from sc2agents.env_wrappers import MoveToBeaconWrapper

__author__ = 'Islam Elnabarawy'

FLAGS = flags.FLAGS


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    FLAGS(sys.argv)

    map_name = 'MoveToBeacon'
    env = MoveToBeaconWrapper()

    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True
    )

    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to {}_model.pkl".format(map_name))
    act.save("{}_model.pkl".format(map_name))

    env.save_replay(map_name)


if __name__ == "__main__":
    main()
