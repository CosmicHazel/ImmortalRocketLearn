# Preliminary setup for serious crowd-sourced model
# Exact setup should probably be in different repo
import os
from typing import Any

import numpy as np
import torch
import torch.jit
import wandb
from redis import Redis
from rlgym.envs import Match
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityReward
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from torch.nn import Linear, Sequential

from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator, RedisRolloutWorker
from rocket_learn.utils.util import SplitLayer
from training.parser import NectoActionTEST


class ExpandAdvancedObs(AdvancedObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return np.expand_dims(obs, 0)


def get_match():
    return Match(
        reward_function=CombinedReward.from_zipped((VelocityPlayerToBallReward(), 0.5),
                                                   (VelocityReward(), 0.5)),
        terminal_conditions=TimeoutCondition(75),
        action_parser=NectoActionTEST(),
        obs_builder=ExpandAdvancedObs(),
        state_setter=DefaultState(),
        self_play=True,
        team_size=1
    )


def make_worker(host, name, limit_threads=True):
    if limit_threads:
        torch.set_num_threads(1)
    r = Redis(host=host, password="rocket-learn")
    return RedisRolloutWorker(r, name, get_match(), past_version_prob=.1, evaluation_prob=0.0).run()


if __name__ == "__main__":
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(project="rocket-learn", entity="cosmicvivacity")

    redis = Redis(password="rocket-learn")
    rollout_gen = RedisRolloutGenerator(redis, save_every=10, logger=logger, act_parse_factory=NectoActionTEST,
                                        obs_build_factory=ExpandAdvancedObs,
                                        rew_func_factory=lambda: CombinedReward.from_zipped(
                                            (VelocityPlayerToBallReward(), 0.5),
                                            (VelocityReward(), 0.5)))

    # jit models can't be pickled
    # ex_inp = (
    #     (torch.zeros((10, 1, 32)), torch.zeros((10, 1 + 6 + 34, 24)), torch.zeros((10, 1 + 6 + 34))),)  # q, kv, mask
    # critic = torch.jit.trace(
    #     func=Necto(EARLPerceiver(128, query_features=32, key_value_features=24), Linear(128, 1)),
    #     example_inputs=ex_inp
    # )
    # actor = torch.jit.trace(
    #     func=Necto(EARLPerceiver(256, query_features=32, key_value_features=24), ControlsPredictorDiscrete(256)),
    #     example_inputs=ex_inp
    # )
    critic = Sequential(Linear(107, 128), Linear(128, 90), Linear(90, 90), Linear(90, 1))
    actor = DiscretePolicy(Sequential(Linear(107, 128), Linear(128, 90), Linear(90, 90), Linear(90, 90), SplitLayer(splits=(90,))))

    lr = 1e-5
    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": lr},
        {"params": critic.parameters(), "lr": lr}
    ])

    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)

    alg = PPO(
        rollout_gen,
        agent,
        ent_coef=0.01,
        n_steps=15_000,
        batch_size=1_500,
        minibatch_size=750,
        epochs=10,
        gamma=599 / 600,  # 5 second horizon
        logger=logger,
    )

    log_dir = "E:\\log_directory\\"
    repo_dir = "E:\\repo_directory\\"

    alg.run(iterations_per_save=10, save_dir="ppos")
