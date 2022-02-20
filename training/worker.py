import sys
from distutils.util import strtobool

import torch
from redis import Redis
from rlgym.envs import Match
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions.common_rewards import VelocityReward, LiuDistancePlayerToBallReward
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym_tools.extra_state_setters.augment_setter import AugmentSetter

from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutWorker
from training.immortalreward import ImmortalReward
from training.learner import WORKER_COUNTER
from training.obs import NectoObsTEST
from training.parser import SetAction
from training.state import ImmortalStateSetter
from training.terminal import ImmortalTerminalCondition
#from training.immortal_obs import ExpandAdvancedObs
from rocket_learn.utils.util import ExpandAdvancedObs


#def get_match(r, force_match_size, replay_arrays, game_speed=100):
from training.test_reward import ImmortalTestReward


def get_match(r, force_match_size, redis, game_speed=100):
    order = (1, 2, 3, 1, 1, 2, 1, 1, 3, 2, 1)  # Close as possible number of agents
    # order = (1, 1, 2, 1, 1, 2, 3, 1, 1, 2, 3)  # Close as possible with 1s >= 2s >= 3s
    # order = (1,)
    team_size = order[r % len(order)]
    if force_match_size:
        team_size = force_match_size

    return Match(

        reward_function=ImmortalTestReward(redis),
        terminal_conditions=TimeoutCondition(75),
        obs_builder=ExpandAdvancedObs(),
        action_parser=SetAction(),
        #state_setter=AugmentSetter(ImmortalStateSetter()),
        state_setter=DefaultState(),
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
    print(r.ping())

    current_prob = .8
    eval_prob = 0.01
    game_speed = 100
    if is_streamer:
        current_prob = 1
        eval_prob = 0
        game_speed = 1

    #try:
        #replay_arrays = _unserialize(r.get("replay-arrays"))
    #except:
        #replay_arrays = []

    return RedisRolloutWorker(r, name,
                              #match=get_match(w, force_match_size, game_speed=game_speed, replay_arrays=replay_arrays),
                              match=get_match(w, force_match_size, r, game_speed=game_speed),
                              current_version_prob=current_prob,
                              evaluation_prob=eval_prob,
                              send_gamestates=send_gamestates,
                              display_only=False)


def main():
    # if not torch.cuda.is_available():
    #     sys.exit("Unable to train on your hardware, perhaps due to out-dated drivers or hardware age.")

    assert len(sys.argv) >= 4  # last is optional to force match size

    force_match_size = 1

    print(len(sys.argv))

    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-n", "--name", required=True)
    # parser.add_argument("--ip", required=True)
    # parser.add_argument("-p", "--password", required=True)
    # parser.add_argument("-c", "--compress", default=True)
    # parser.add_argument("-d", "--display-only", default=True)
    # parser.add_argument("-g", "--gamemode", default=None)
    # parser.add_argument("-p", "--password", required=True)
    # parser.add_argument("-p", "--password", required=True)
    #
    # compress = bool(strtobool(sys.argv[4])) if len(sys.argv) >= 5 else True
    # display_only = bool(strtobool(sys.argv[5])) if len(sys.argv) >= 6 else False
    # force_match_size = int(sys.argv[6]) if len(sys.argv) >= 7 else None
    # current_version_prob = float(sys.argv[7]) if len(sys.argv) >= 8 else 0.8
    # evaluation_prob = float(sys.argv[8]) if len(sys.argv) >= 9 else 0.01

    if len(sys.argv) == 5:
        _, name, ip, password, compress = sys.argv
        stream_state = False
    elif len(sys.argv) == 6:
        _, name, ip, password, compress, is_stream = sys.argv

        # atm, adding an extra arg assumes you're trying to stream
        stream_state = True
        force_match_size = int(1)

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
