import random
from abc import ABC, abstractmethod

from utils.constant import *

class BaseAgent(ABC):
    """
    Base class for the agents
    """

    def __init__(self, config, possible_actions=[], name=None, **kwargs):
        self._type = config[MODEL][AGENT]
        self.actions = possible_actions
        self.gamma = config[MODEL][GAMMA]
        self._lambda = config[MODEL][LAMBDA]
        self.save_dir = config[MODEL][SAVE_DIR]
        self.load_path = config[MODEL][LOAD_PATH]
        self.num_optimisers = config[MODEL][NUM_OPTIMIZERS]
        if name:
            self.name = name
        else:
            self.name = ALICE
        if(self.num_optimisers==1):
            self.learning_rate = config[MODEL][LEARNING_RATE]
        elif(self.num_optimisers==2):
            self.learning_rate_actor = config[MODEL][LEARNING_RATE_ACTOR]
            self.learning_rate_critic = config[MODEL][LEARNING_RATE_CRITIC]

    #   cant use "lambda" as it is a keyword in python

    @abstractmethod
    def get_action(self, observation):
        pass

    def get_random_action(self):
        """
        Return a random action
        """
        return random.choice(self.actions)

    def get_optimisers(self, optimiser_name):
        return None

    def update_policy(self, optimizer, **kwargs):
        return optimizer

    def set_initial_state(self):
        pass
