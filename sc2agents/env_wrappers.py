import logging

import gym
# noinspection PyUnresolvedReferences
import sc2gym.envs
from gym import spaces
from pysc2.lib import actions, features

__author__ = 'Islam Elnabarawy'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1

_NO_OP = actions.FUNCTIONS.no_op.id

_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_NOT_QUEUED = [0]

_ACTION_OFFSET = {1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}


class MoveToBeaconWrapper(object):
    def __init__(self, visualize=False):
        self._env = gym.make("SC2MoveToBeacon-v0")
        self._env.visualize = visualize
        screen_shape = self._env.observation_spec["screen"]
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=4, shape=screen_shape)
        self.last_observation = None

    def reset(self):
        self._env.reset()
        obs, reward, done, info = self._env.step([_SELECT_ARMY, _SELECT_ALL])
        self.last_observation = obs
        obs = self._extract_observation(obs)
        return obs

    def step(self, action):
        action = self._translate_action(action)
        obs, reward, done, info = self._env.step(action)
        if obs is None:
            return None, 0, True, {}
        self.last_observation = obs
        obs = self._extract_observation(obs)
        return obs, reward, done, info

    def _extract_observation(self, obs):
        return obs.observation["screen"]

    def _translate_action(self, action):
        if action == 0:
            return [_NO_OP]
        player_relative = self._extract_observation(self.last_observation)[_PLAYER_RELATIVE]
        player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
        if not player_y.any() or not player_x.any():
            return [_NO_OP]
        player = (int(player_x.mean()), int(player_y.mean()))
        offset = 8
        movement = _ACTION_OFFSET[action]
        target = [0, 0]
        for i in range(2):
            target[i] = player[i] + movement[i] * offset
            if target[i] < 0:
                target[i] = 0
            if target[i] >= player_relative.shape[i]:
                target[i] = player_relative.shape[i] - 1
        return [_MOVE_SCREEN, _NOT_QUEUED, target]

    def save_replay(self, replay_dir):
        return self._env.save_replay(replay_dir)
