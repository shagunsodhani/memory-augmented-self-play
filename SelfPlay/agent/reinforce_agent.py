from agent.base_agent import BaseAgent
from policy.registry import choose_policy
from utils.constant import MODEL, USE_BASELINE, BATCH_SIZE, ENV, IS_SELF_PLAY, \
    IS_SELF_PLAY_WITH_MEMORY, EPISODE_MEMORY_SIZE
from utils.optim_registry import choose_optimiser


class ReinforceAgent(BaseAgent):
    """
    An agent trained using REINFORCE-Baseline algorithm
    """

    def __init__(self, config, possible_actions=[], name=None, input_size=None):
        super(ReinforceAgent, self).__init__(config=config,
                                             possible_actions=possible_actions, name=name)
        self.policy = choose_policy(env_name=config[MODEL][ENV],
                                    agent_type=self._type,
                                    use_baseline=config[MODEL][USE_BASELINE],
                                    input_size=input_size,
                                    num_actions=len(self.actions),
                                    batch_size=config[MODEL][BATCH_SIZE],
                                    is_self_play=config[MODEL][IS_SELF_PLAY],
                                    is_self_play_with_memory=config[MODEL][IS_SELF_PLAY_WITH_MEMORY],
                                    _lambda=self._lambda,
                                    episode_memory_size=config[MODEL][EPISODE_MEMORY_SIZE])

    def get_optimisers(self, optimiser_name):
        optimiser = choose_optimiser(optimiser_name=optimiser_name)
        params = self.policy.memory.get_params()
        if (self.num_optimisers == 1):
            params += self.policy.parameters()
            return (optimiser(params, self.learning_rate),)
            # elif (self.num_optimisers == 2):
            #     return (optimiser(self.policy.get_actor_params(), self.learning_rate_actor),
            #             optimiser(self.policy.get_critic_params(), self.learning_rate_critic))

    def update_policy(self, optimizers, observation=None):
        # Only to be called once the episode is over
        return self.policy.update(optimisers=optimizers, gamma=self.gamma, agent_name=self.name)

    def get_action(self, observation):
        return self.actions[self.policy.get_action(observation)]

    def save_model(self, epochs, optimisers, name, timestamp):
        return self.policy.save_model(epochs=epochs, optimisers=optimisers,
                                      save_dir=self.save_dir, name=name, timestamp=timestamp)

    def load_model(self, optimisers, name, timestamp):
        return self.policy.load_model(optimisers=optimisers,
                                      load_path=self.load_path, name=name, timestamp=timestamp)
