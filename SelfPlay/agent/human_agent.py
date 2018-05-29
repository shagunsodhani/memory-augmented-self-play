from agent.base_agent import BaseAgent


class HumanAgent(BaseAgent):
    """
    An agent controlled by human
    """

    def __init__(self, config, possible_actions=[], name=None, **kwargs):
        super(HumanAgent, self).__init__(config, possible_actions=possible_actions, name=name, **kwargs)

    def get_action(self, observation):
        """
        This code is borrowed from:
        https://github.com/facebook/MazeBase/blob/23454fe092ecf35a8aab4da4972f231c6458209b/py/example.py#L172
        """
        print(list(enumerate(self.actions)))
        action_index = -1
        while action_index not in range(len(self.actions)):
            action_index = input("Input a number to choose the action: ")
            try:
                action_index = int(action_index)
            except ValueError:
                action_index = -1
        return self.actions[action_index]
