import numpy as np

class Observation():
    def __init__(self, id=-1, reward=0, state=None, is_episode_over=False):
        self.id = id
        self.reward = reward
        self.state = state
        self.is_episode_over = is_episode_over

    def are_equal(self, other, are_states_equal):
        if (self.id != other.id):
            return False
        elif (self.reward != other.reward):
            return False
        elif (not are_states_equal(self.state, other.state)):
            return False
        elif (self.is_episode_over != other.is_episode_over):
            return False
        return True


class ObservationTuple():
    def __init__(self, start=None, end=None, target=None, memory = None):
        self.start = start
        self.end = end
        self.target = target
        self.memory = memory
