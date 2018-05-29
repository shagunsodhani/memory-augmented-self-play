from utils.constant import *


class MemoryConfig:
    def __init__(self, episode_memory_size=10, input_dim=156*10*10*2, output_dim=50):
        self.data = {
            EPISODE_MEMORY_SIZE: episode_memory_size,
            INPUT_DIM: input_dim,
            OUTPUT_DIM: output_dim
        }

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value