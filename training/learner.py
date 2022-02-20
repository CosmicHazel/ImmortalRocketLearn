import os
import sys

import torch
import wandb
from redis import Redis
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.reward_functions.common_rewards import LiuDistancePlayerToBallReward

from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.utils.util import ExpandAdvancedObs
from training.agent import get_agent
from training.immortalreward import ImmortalReward
from training.obs import NectoObsTEST
from training.parser import SetAction
import os

from training.test_reward import ImmortalTestReward

WORKER_COUNTER = "worker-counter"

config = dict(
    seed=123,
    actor_lr=1e-5,
    critic_lr=1e-5,
    n_steps=15_000,
    batch_size=1_500,
    minibatch_size=750,
    epochs=20,
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


    run_id = None
    #run_id = "amixi1oi"

    _, ip, password = sys.argv
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(project="Immortal1v1", entity=os.environ["entity"], id=run_id, config=config)
    torch.manual_seed(logger.config.seed)

    redis = Redis(host=ip, password=password)
    redis.delete(WORKER_COUNTER)  # Reset to 0

    rollout_gen = RedisRolloutGenerator(redis, ExpandAdvancedObs, lambda: ImmortalTestReward(redis, silent=True), SetAction,
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
    #alg.load("ppos/Immortal1v1_1645339021.8435597/Immortal1v1_640/checkpoint.pt")
    # alg.agent.optimizer.param_groups[0]["lr"] = logger.config.actor_lr
    # alg.agent.optimizer.param_groups[1]["lr"] = logger.config.critic_lr

    log_dir = "E:\\log_directory\\"
    repo_dir = "E:\\repo_directory\\"

    alg.run(iterations_per_save=logger.config.iterations_per_save, save_dir="ppos")
