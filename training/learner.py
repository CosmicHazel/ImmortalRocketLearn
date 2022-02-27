import os
import sys

import torch
import wandb

from redis import Redis
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityReward, EventReward, \
    VelocityBallToGoalReward

from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.utils.util import ExpandAdvancedObs
from training.agent import get_agent
from training.parser import ImmortalAction

WORKER_COUNTER = "worker-counter"

import numpy as np


def get_latest_checkpoint():
    subdir = 'ppos'

    all_subdirs = [os.path.join(subdir, d) for d in os.listdir(subdir) if os.path.isdir(os.path.join(subdir, d))]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    all_subdirs = [os.path.join(latest_subdir, d) for d in os.listdir(latest_subdir) if
                   os.path.isdir(os.path.join(latest_subdir, d))]
    latest_subdir = (max(all_subdirs, key=os.path.getmtime))
    full_dir = os.path.join(latest_subdir, 'checkpoint.pt')
    print(full_dir)

    return full_dir


if __name__ == "__main__":
    from rocket_learn.ppo import PPO

    frame_skip = 8  # Number of ticks to repeat an action
    half_life_seconds = 5  # Easier to conceptualize, after this many seconds the reward discount is 0.5
    run_id = "39riebfz"
    clear = False
    file = get_latest_checkpoint()
    #file = "ppos\Immortal_1645886051.467028\Immortal_65/checkpoint.pt"

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))

    config = dict(
        seed=125,
        actor_lr=3e-4,
        critic_lr=3e-4,
        n_steps=200_000,
        batch_size=40_000,
        minibatch_size=20_000,
        epochs=32,
        gamma=gamma,
        iterations_per_save=5
    )

    print(gamma)

    _, ip, password = sys.argv
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(project="Immortal", entity="cosmicvivacity", id=run_id, config=config)
    torch.manual_seed(logger.config.seed)

    redis = Redis(host=ip, password=password)
    redis.delete(WORKER_COUNTER)  # Reset to 0

    rollout_gen = RedisRolloutGenerator(redis, ExpandAdvancedObs,
                                        lambda: CombinedReward.from_zipped((VelocityPlayerToBallReward(), 0.4),
                                                                           (VelocityReward(), 0.6),
                                                                           (VelocityBallToGoalReward(), 2.0),
                                                                           EventReward(team_goal=100, save=30, demo=20,
                                                                                       concede=-100),
                                                                           ), ImmortalAction,
                                        save_every=logger.config.iterations_per_save,
                                        logger=logger, clear=clear)

    agent = get_agent(actor_lr=logger.config.actor_lr, critic_lr=logger.config.critic_lr)

    alg = PPO(
        rollout_gen,
        agent,
        n_steps=logger.config.n_steps,
        batch_size=logger.config.batch_size,
        minibatch_size=logger.config.minibatch_size,
        epochs=logger.config.epochs,
        gamma=logger.config.gamma,
        logger=logger,
    )


    # TODO fix empty folders
    def get_latest_checkpoint():
        subdir = 'ppos'

        all_subdirs = [os.path.join(subdir, d) for d in os.listdir(subdir) if os.path.isdir(os.path.join(subdir, d))]
        latest_subdir = max(all_subdirs, key=os.path.getmtime)
        all_subdirs = [os.path.join(latest_subdir, d) for d in os.listdir(latest_subdir) if
                       os.path.isdir(os.path.join(latest_subdir, d))]
        latest_subdir = (max(all_subdirs, key=os.path.getmtime))
        full_dir = os.path.join(latest_subdir, 'checkpoint.pt')
        print(full_dir)

        return full_dir


    if file:
        alg.load(file, continue_iterations=not clear)
        # alg.load(get_latest_checkpoint())
        # alg.agent.optimizer.param_groups[0]["lr"] = logger.config.actor_lr
        # alg.agent.optimizer.param_groups[1]["lr"] = logger.config.critic_lr

    log_dir = "E:\\log_directory\\"
    repo_dir = "E:\\repo_directory\\"

    alg.run(iterations_per_save=logger.config.iterations_per_save, save_dir="ppos")
