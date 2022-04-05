import sys
from distutils.util import strtobool

import torch
from redis import Redis
from rlgym.envs import Match
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, \
    VelocityBallToGoalReward, EventReward, VelocityReward
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition, \
    NoTouchTimeoutCondition

from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutWorker
from rocket_learn.utils.util import ExpandAdvancedObs
from training.learner import WORKER_COUNTER
from training.parser import ImmortalAction
from training.state import ImmortalStateSetter


def get_match(r, force_match_size, game_speed=100):
    team_size = force_match_size
    frame_skip = 8  # Number of ticks to repeat an action
    fps = 120 / frame_skip

    return Match(
        reward_function=CombinedReward.from_zipped((VelocityPlayerToBallReward(), 0.4), (VelocityReward(), 0.6),
                                                   (VelocityBallToGoalReward(), 2.0),
                                                   EventReward(team_goal=1200, save=200, demo=500,
                                                               concede=-1000),
                                                   ),
        terminal_conditions=[TimeoutCondition(round(fps * 30)), NoTouchTimeoutCondition(round(fps * 20)),
                             GoalScoredCondition()],
        obs_builder=ExpandAdvancedObs(),
        action_parser=ImmortalAction(),
        state_setter=ImmortalStateSetter(),
        self_play=True,
        team_size=team_size,
        game_speed=game_speed,
    )


def make_worker(host, name, password, limit_threads=True, send_gamestates=False, force_match_size=None,
                is_streamer=False):
    if limit_threads:
        torch.set_num_threads(1)
    r = Redis(host=host, password=password)
    w = r.incr(WORKER_COUNTER) - 1

    current_prob = .8
    eval_prob = 0.00
    game_speed = 100
    if is_streamer:
        current_prob = 1
        eval_prob = 0
        game_speed = 1

    #    replay_arrays = _unserialize(r.get("replay-arrays"))

    return RedisRolloutWorker(r, name,
                              match=get_match(w, force_match_size, game_speed=game_speed),
                              # replay_arrays=replay_arrays),
                              current_version_prob=current_prob,
                              #past_version_prob=1-current_prob,
                              evaluation_prob=eval_prob,
                              send_gamestates=send_gamestates#,
                              #display_only=False
                              )


def main():
    assert len(sys.argv) >= 4  # last is optional to force match size

    force_match_size = 1

    print(len(sys.argv))

    if len(sys.argv) == 5:
        _, name, ip, password, compress = sys.argv
        stream_state = False
    elif len(sys.argv) == 6:
        _, name, ip, password, compress, is_stream = sys.argv

        # atm, adding an extra arg assumes you're trying to stream
        stream_state = True
        force_match_size = int(2)

    elif len(sys.argv) == 7:
        _, name, ip, password, compress, is_stream, force_match_size = sys.argv

        # atm, adding an extra arg assumes you're trying to stream
        stream_state = True
        force_match_size = int(force_match_size)

        if not (1 <= force_match_size <= 3):
            force_match_size = None
    else:
        raise ValueError

    try:
        worker = make_worker(ip, name, password,
                             limit_threads=True,
                             send_gamestates=bool(strtobool(compress)),
                             force_match_size=force_match_size,
                             is_streamer=stream_state)
        worker.run()
    finally:
        print("Problem Detected. Killing Worker...")


if __name__ == '__main__':
    main()
