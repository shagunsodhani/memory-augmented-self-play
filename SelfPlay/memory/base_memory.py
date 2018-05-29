from collections import deque

import numpy as np
import torch.nn as nn

from model.base_model import BaseModel
from memory.memory_config import MemoryConfig
from utils.constant import *
import torch
from torch.autograd import Variable


class BaseMemory(BaseModel):
    def __init__(self, memory_config):
        super(BaseMemory, self).__init__()
        self.memory_config = memory_config
        self.internal_memory = deque(maxlen=memory_config[EPISODE_MEMORY_SIZE])
        self.hidden_state = None
        self.reset()

        # response_network is the network that map the start and the end observations into a single vector
        self.response_network = self.get_response_network()
        self.summary_network = self.get_summary_network()
        self.init_weights()


    def reset(self):
        self.internal_memory.clear()
        self.hidden_state = Variable(torch.from_numpy(np.zeros(shape=(self.memory_config[INPUT_DIM]))
                                                      )).float()

    # def get_response_network(self):
    #     return nn.Sequential(
    #         nn.Linear(self.memory_config[INPUT_DIM], self.memory_config[OUTPUT_DIM])
    #     )

    def get_response_network(self):
        def _reponse():
            return self.hidden_state
        return _reponse

    def get_summary_network(self):
        def _summariser():
            num_entries = len(self.internal_memory)
            if(num_entries ==0 ):
                return self.hidden_state
            else:
                return (sum(self.internal_memory)/num_entries).squeeze(0)
        return _summariser

    def update_memory(self, history):
        self.internal_memory.append(history)
        # self.internal_memory.append(Variable(torch.from_numpy(history)).float())

    def summarize(self):
        # This is the method which the different classes would override
        return self.forward()

    def forward(self):
        self.hidden_state = self.summary_network()
        return self.response_network()

    def init_weights(self):
        self.init_weights_response_network()
        self.init_weights_summary_network()

    def init_weights_response_network(self):
        # based on https://discuss.pytorch.org/t/how-to-initialize-weights-in-nn-sequential-container/8534/3
        self.apply(self._init_weights)

    def init_weights_summary_network(self):
        pass

    def get_params(self):
        params = []
        for param in self.named_parameters():
            if param[1].requires_grad:
                params.append(param[1])
        return params

    def _init_weights(self, module):
        if type(module) == nn.Linear:
            module.weight.data.fill_(1.0)

if __name__ == '__main__':
    memory = BaseMemory(memory_config=MemoryConfig())
    print(memory)
