import random

from environment.env import Environment
from environment.mazebase_wrapper import MazebaseWrapper
from environment.observation import ObservationTuple, Observation
from utils.constant import *
from copy import deepcopy
from environment.selfplay import SelfPlay

class SelfPlayMemory(SelfPlay):
    """
    Wrapper class over self play environment
    """

    def __init__(self, environment, task=COPY):
        super(SelfPlayMemory, self).__init__(environment=environment, task=task)
        self.environment = environment
        self.name = SELFPLAY + "_" + MEMORY + "_" + self.environment.name
        # The environment (along with the state) in which alice starts

    def observe(self):
        self._process_observation()
        return self.observation

    def alice_observe(self):
        observation = self.observe()
        return (observation, self.alice_observations.start)

if __name__ == "__main__":
    play = SelfPlayMemory(environment=MazebaseWrapper())
    actions = play.all_possible_actions()
    print(actions)
    for i in range(100):
        print("==============")
        _action = random.choice(actions)
        print(_action)
        play.act(_action)
        print((play.observe()).reward)
