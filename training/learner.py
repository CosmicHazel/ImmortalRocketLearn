import os
import sys

import torch
import wandb
from redis import Redis
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityReward

from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.utils.util import ExpandAdvancedObs
from training.agent import get_agent
from training.parser import ImmortalAction

WORKER_COUNTER = "worker-counter"

config = dict(
    seed=123,
    actor_lr=1e-5,
    critic_lr=1e-5,
    n_steps=150_000,
    batch_size=50_000,
    minibatch_size=25_000,
    epochs=20,
    gamma=0.995,
    iterations_per_save=10
)

if __name__ == "__main__":
    from rocket_learn.ppo import PPO

    run_id = "dyemrdlu"

    _, ip, password = sys.argv
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(project="Immortal", entity="cosmicvivacity", id=run_id, config=config)
    torch.manual_seed(logger.config.seed)

    redis = Redis(host=ip, password=password)
    redis.delete(WORKER_COUNTER)  # Reset to 0

    rollout_gen = RedisRolloutGenerator(redis, ExpandAdvancedObs,
                                        lambda: CombinedReward.from_zipped((VelocityPlayerToBallReward(), 1.0),
                                                                           ), ImmortalAction,
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

    #if run_id is not None:
    #alg.load("ppos/Immortal_1645428859.310328/Immortal_1860/checkpoint.pt")
    alg.load(get_latest_checkpoint())
        # alg.agent.optimizer.param_groups[0]["lr"] = logger.config.actor_lr
        # alg.agent.optimizer.param_groups[1]["lr"] = logger.config.critic_lr

    log_dir = "E:\\log_directory\\"
    repo_dir = "E:\\repo_directory\\"

    alg.run(iterations_per_save=logger.config.iterations_per_save, save_dir="ppos")
