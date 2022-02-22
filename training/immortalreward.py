from redis.client import Redis
from rlgym.utils import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.reward_functions.common_rewards import VelocityReward, EventReward, LiuDistancePlayerToBallReward, \
    VelocityPlayerToBallReward, VelocityBallToGoalReward
import numpy as np
from rlgym_tools.extra_rewards.diff_reward import DiffReward


class ImmortalReward(RewardFunction):
    # Simple reward function to ensure the model is training.
    def __init__(self, redis, silent=False):
        super().__init__()

        self.silent = silent
        self.redis = redis

        self.vel_reward = 0
        self.event_reward = 0
        # self.dtb_reward = 0
        self.vtb_reward = 0
        self.vtg_reward = 0

        self.starting_n_updates = 0.0

        self.vel_reward_weight = 0.66
        self.touch_reward_weight = 100.0
        self.goal_reward_weight = 1000.0
        # self.n_dtb_reward = 1.0
        self.vtb_reward_weight = 0.33
        self.vtg_reward_weight = 1.0

        self.event_reward_object = EventReward(touch=self.touch_reward_weight, team_goal=self.goal_reward_weight,
                                               concede=-self.goal_reward_weight)
        # self.distance_to_ball_reward_object = LiuDistancePlayerToBallReward()
        self.vel_reward_object = VelocityReward()
        self.vtb_reward_object = VelocityPlayerToBallReward()
        self.vtg_reward_object = VelocityBallToGoalReward()

    def _update_reward(self, starting_reward, max_reward, time_steps_to_reach_max=50_000_000.0) -> float:
        starting_reward_amount = float(starting_reward)
        max_reward = float(max_reward)
        time_steps_to_reach_max = float(time_steps_to_reach_max)
        n_steps = 200_000.0
        starting_n_updates = self.starting_n_updates
        inc_reward_total = max_reward - starting_reward_amount
        percent_inc_per_iteration = n_steps / time_steps_to_reach_max
        inc_reward_per_iteration = inc_reward_total * percent_inc_per_iteration
        num_iterations = time_steps_to_reach_max / n_steps

        n_updates = int(self.redis.get("num-updates"))
        inc_updates = float(n_updates - starting_n_updates)
        if inc_updates > num_iterations:
            return max_reward
        elif inc_updates > 0:
            return inc_updates * inc_reward_per_iteration + starting_reward_amount
        return starting_reward_amount

    def reset(self, initial_state: GameState):

        if not self.silent:
            print(f"Total event reward for episode: {self.event_reward}")
            print(f"Total velocity reward for episode: {self.vel_reward}")
            #print(f"Total distance to ball reward for episode: {self.dtb_reward}")
            print(f"Total velocity to ball reward for episode: {self.vtb_reward}")
            print(f"Total velocity to goal reward for episode: {self.vtg_reward}")

        self.event_reward = 0
        self.vel_reward = 0
        #self.dtb_reward = 0
        self.vtb_reward = 0
        self.vtg_reward = 0

        self.vel_reward_object.reset(initial_state=initial_state)
        self.event_reward_object.reset(initial_state=initial_state)
        #self.distance_to_ball_reward_object.reset(initial_state=initial_state)
        self.vtb_reward_object.reset(initial_state=initial_state)
        self.vtg_reward_object.reset(initial_state=initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        this_reward = 0.0

        this_reward += self.vel_reward_object.get_reward(player=player, state=state, previous_action=previous_action)
        self.vel_reward += this_reward

        this_event_reward = self.event_reward_object.get_reward(player=player, state=state,
                                                                previous_action=previous_action)

        if this_event_reward > 0:
            self.event_reward += this_event_reward
        this_reward += this_event_reward

        #this_dtb_reward = self.n_dtb_reward * self.distance_to_ball_reward_object.get_reward(player=player, state=state,
        #                                                                                     previous_action=previous_action)
        #self.dtb_reward += this_dtb_reward
        #this_reward += this_dtb_reward

        this_vtb_reward = self.vtb_reward_weight * self.vtb_reward_object.get_reward(player=player, state=state,
                                                                                     previous_action=previous_action)
        self.vtb_reward += this_vtb_reward
        this_reward += this_vtb_reward

        this_vtg_reward = self.vtg_reward_weight * self.vtg_reward_object.get_reward(player=player, state=state,
                                                                                     previous_action=previous_action)
        if this_vtg_reward > 0:
            self.vtg_reward += this_vtg_reward
        this_reward += this_vtg_reward

        return this_reward


if __name__ == '__main__':
    # ap = NectoActionTEST()
    ap = ImmortalReward(redis=Redis(host="localhost", password="rocket-learn"))
    print("worked")
