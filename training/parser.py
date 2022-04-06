from typing import Any

import gym.spaces
import numpy as np
from gym.spaces import Discrete
from rlgym.utils.action_parsers import ActionParser, DiscreteAction
from rlgym.utils.gamestates import GameState


class ImmortalAction(DiscreteAction):
    def get_action_space(self) -> gym.spaces.Space:
        # throttle/pitch, steer/pitch/roll, jump, boost and handbrake/air roll
        return gym.spaces.MultiDiscrete([self._n_bins] * 3 + [2] * 3)

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        #actions = actions.reshape((-1, 8))
        actions[:, 0] = actions[:, 0] / (self._n_bins // 2) - 1
        actions[:, 1] = actions[:, 1] / (self._n_bins // 2) - 1
        actions[:, 2] = actions[:, 2] / (self._n_bins // 2) - 1

        parsed = np.zeros((actions.shape[0], 8))
        parsed[:, 0] = actions[:, 0]  # throttle
        parsed[:, 1] = actions[:, 1]  # steer
        parsed[:, 2] = actions[:, 0]  # pitch
        parsed[:, 3] = actions[:, 1]  # yaw
        parsed[:, 4] = actions[:, 2]  # roll
        parsed[:, 5] = actions[:, 3]  # jump
        parsed[:, 6] = actions[:, 4]  # boost
        parsed[:, 7] = actions[:, 5]  # handbrake

        return parsed


SetAction = ImmortalAction


class NectoActionTEST(ActionParser):
    def __init__(self):
        super().__init__()
        self._lookup_table = self._make_lookup_table()

    @staticmethod
    def _make_lookup_table():
        actions = []
        # Ground
        for throttle in (-1, 0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and roll != 0:  # Only need yaw for sideflip
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            # Enable handbrake for potential wavedashes
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
        actions = np.array(actions)
        return actions

    def get_action_space(self) -> gym.spaces.Space:
        return Discrete(len(self._lookup_table))

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        return self._lookup_table[actions]


if __name__ == '__main__':
    ap = NectoActionTEST()
    print(ap.get_action_space())
