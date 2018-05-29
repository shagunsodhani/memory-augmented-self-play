from utils.constant import *


class PolicyConfig:
    def __init__(self, env_name = ENVIRONMENT,
                 agent_type = REINFORCE_AGENT,
                 use_baseline=True,
                 input_size=None,
                 num_actions=10,
                 batch_size=32,
                 is_self_play=False,
                 is_self_play_with_memory=False,
                 _lambda=0.3,
                 agent=AGENT,
                 episode_memory_size=10,
                 memory_type=BASE_MEMORY):
        self.data = {
            ENVIRONMENT: env_name,
            AGENT_TYPE: agent_type,
            USE_BASELINE: use_baseline,
            INPUT_SIZE: input_size,
            NUM_ACTIONS: num_actions,
            BATCH_SIZE: batch_size,
            IS_SELF_PLAY: is_self_play,
            IS_SELF_PLAY_WITH_MEMORY: is_self_play_with_memory,
            LAMBDA: _lambda,
            AGENT: agent,
            EPISODE_MEMORY_SIZE: episode_memory_size,
            MEMORY_TYPE: memory_type,
        }

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value