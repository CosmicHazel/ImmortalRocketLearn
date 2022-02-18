import os
import sys

import torch
import wandb
from redis import Redis
from rlgym.utils.reward_functions.common_rewards import LiuDistancePlayerToBallReward

from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator
from training.agent import get_agent
from training.immortalreward import ImmortalReward
from training.obs import NectoObsTEST
from training.parser import SetAction
from training.reward import NectoRewardFunction
import os

WORKER_COUNTER = "worker-counter"

config = dict(
    seed=1234,
    actor_lr=1e-5,
    critic_lr=1e-5,
    n_steps=500_000,
    batch_size=100_000,
    minibatch_size=50_000,
    epochs=30,
    gamma=0.995,
    iterations_per_save=5
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


if __name__ == "__main__":
    from rocket_learn.ppo import PPO


    #run_id = None
    run_id = "3f8f4ak6"

    _, ip, password = sys.argv
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(project="rocket-learn", entity=os.environ["entity"], id=run_id, config=config)
    torch.manual_seed(logger.config.seed)

    redis = Redis(host=ip, password=password)
    redis.delete(WORKER_COUNTER)  # Reset to 0

    rollout_gen = RedisRolloutGenerator(redis, lambda: NectoObsTEST(6), ImmortalReward, SetAction,
                                        save_every=logger.config.iterations_per_save,
                                        logger=logger, clear=run_id is None)

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

    #if run_id is not None:
    #alg.load(get_latest_checkpoint())
    alg.load("ppos/rocket-learn_1645136428.1678982/rocket-learn_9070/checkpoint.pt")
    # alg.agent.optimizer.param_groups[0]["lr"] = logger.config.actor_lr
    # alg.agent.optimizer.param_groups[1]["lr"] = logger.config.critic_lr

    log_dir = "E:\\log_directory\\"
    repo_dir = "E:\\repo_directory\\"

    alg.run(iterations_per_save=logger.config.iterations_per_save, save_dir="ppos")
