from agent.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """
    A randomly behaving agent
    """

    def __init__(self, config, possible_actions=[], name=None, **kwargs):
        super(RandomAgent, self).__init__(config, possible_actions=possible_actions, name=name, **kwargs)

    def get_action(self, observation):
        """Return a random action"""
        return self.get_random_action()

