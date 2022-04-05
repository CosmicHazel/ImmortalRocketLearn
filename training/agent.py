import torch
from torch.nn import Linear, Sequential, LeakyReLU

from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.utils.util import SplitLayer


def get_critic():
    return Sequential(
        Linear(107, 512), LeakyReLU(),
        Linear(512, 512), LeakyReLU(),
        Linear(512, 512), LeakyReLU(),
        Linear(512, 512), LeakyReLU(),
        Linear(512, 1))


def get_actor():
    split = (3, 3, 3, 2, 2, 2)
    return DiscretePolicy(Sequential(Linear(107, 512), LeakyReLU(),
                                     Linear(512, 512), LeakyReLU(),
                                     Linear(512, 512), LeakyReLU(),
                                     Linear(512, 512), LeakyReLU(),
                                     Linear(512, 15),
                                     SplitLayer(splits=split)))

def get_agent(actor_lr, critic_lr=None):
    actor = get_actor()
    critic = get_critic()
    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": actor_lr},
        {"params": critic.parameters(), "lr": critic_lr if critic_lr is not None else actor_lr}
    ])

    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)
    return agent
