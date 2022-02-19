from rlgym.utils import TerminalCondition
from rlgym.utils.gamestates import GameState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition


class ImmortalTerminalCondition(TerminalCondition):
    """
    A condition that will terminate an episode after some number of steps.
    """
    goal_condition: GoalScoredCondition

    def __init__(self, redis):
        super().__init__()
        self.redis = redis
        seconds = self._update_seconds()
        
        self.goal_condition = GoalScoredCondition()

        print(seconds)

        self.tick_skip = 8.0
        self.max_steps = round(seconds * 120 / self.tick_skip)
        self.steps = 0

    def _update_seconds(self) -> float:
        # seconds = 4.0
        # n_updates = int(self.redis.get("num-updates"))
        # inc_updates = float(n_updates - 2199)
        # if inc_updates > 120:
        #     return 10.0
        # elif inc_updates > 0:
        #     seconds += inc_updates / 20.0
        # return seconds
        return 10.0

    def reset(self, initial_state: GameState):
        """
        Reset the step counter.
        """
        seconds = self._update_seconds()

        print(seconds)

        self.max_steps = round(seconds * 120 / self.tick_skip)
        self.steps = 0

        self.goal_condition.reset(initial_state=initial_state)

    def is_terminal(self, current_state: GameState) -> bool:

        """
        Increment the current step counter and return `True` if `max_steps` have passed.
        """

        self.steps += 1
        return (self.steps >= self.max_steps) or self.goal_condition.is_terminal(current_state)
        