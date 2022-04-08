import os
import wandb
import numpy
from typing import Any

import torch.jit
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from torch.nn import Linear, Sequential

from redis import Redis

from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions.default_reward import DefaultReward
from rlgym.utils.action_parsers.discrete_act import DiscreteAction

from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.utils.util import SplitLayer


# rocket-learn always expects a batch dimension in the built observation
from training.parser import ImmortalAction


class ExpandAdvancedObs(AdvancedObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)


if __name__ == "__main__":
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(project="demo", entity="cosmicvivacity")
    logger.name = "DEFAULT_LEARNER_EXAMPLE"

    redis = Redis(password="")


    def obs():
        return ExpandAdvancedObs()

    def rew():
        return DefaultReward()

    def act():
        return KBMAction()

    # SPECIFIES HOW OFTEN OLD VERSIONS ARE SAVED TO REDIS
    rollout_gen = RedisRolloutGenerator(redis, obs, rew, act,
                                        logger=logger, save_every=50000)

    critic = Sequential(Linear(107, 128), Linear(128, 64), Linear(64, 32), Linear(32, 1))
    split = (3, 3, 2, 2, 2)
    actor = DiscretePolicy(
        Sequential(Linear(107, 128), Linear(128, 64), Linear(64, 32), Linear(32, 12), SplitLayer(splits=split)), split)

    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": 5e-5},
        {"params": critic.parameters(), "lr": 5e-5}
    ])

    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)

    alg = PPO(
        rollout_gen,
        agent,
        ent_coef=0.01,
        n_steps=1_000_000,
        batch_size=20_000,
        minibatch_size=10_000,
        epochs=10,
        gamma=599 / 600,
        logger=logger,
    )

    #SPECIFIES HOW OFTEN CHECKPOINTS ARE SAVED
    alg.run(iterations_per_save=10, save_dir="checkpoint_save_directory")
