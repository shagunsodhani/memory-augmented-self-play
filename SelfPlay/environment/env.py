from abc import ABC, abstractmethod
from copy import deepcopy

class Environment(ABC):
    """
    Base class for the environments
    """

    def __init__(self):
        self.name = None
        pass

    @abstractmethod
    def observe(self):
        '''Return an object of class environment.observation'''
        pass

    @abstractmethod
    def reset(self):
        '''Return an object of class environment.observation'''
        pass

    @abstractmethod
    def display(self):
        '''Prints the environment on the screen and does not return anything'''
        pass

    @abstractmethod
    def is_over(self):
        '''Return a boolean'''
        pass

    @abstractmethod
    def act(self, action):
        '''Return an object of class environment.observation'''
        pass

    @abstractmethod
    def all_possible_actions(self):
        '''Return a list of possible actions(ints)'''
        pass

    @abstractmethod
    def set_seed(self, seed):
        '''Method to set the seed for the environment'''
        pass

    def are_states_equal(self, state_1, state_2):
        '''Method to compare if two states are equal or sufficiently close'''
        return state_1 == state_2

    def are_observations_equal(self, obs1, obs2):
        '''Method to compare if two states observations are equal or "sufficiently" close'''
        return obs1.are_equal(self, other = obs2, are_states_equal = self.are_states_equal)

    def validate_action(self, action):
        '''Method to check if an action is supported'''
        if action not in self.all_possible_actions():
            raise Exception("Invalid action ({}) being passed. Only following actions are supported: ({})\n"
                            .format(action, ", ".join(self.all_possible_actions())))

    def create_copy(self):
        return deepcopy(self)

    def load_copy(self, env_copy):
        self = deepcopy(env_copy)

    
