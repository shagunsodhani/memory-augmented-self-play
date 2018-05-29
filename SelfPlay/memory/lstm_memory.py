from collections import deque

import numpy as np
import torch.nn as nn

from memory.base_memory import BaseMemory
from memory.memory_config import MemoryConfig
from utils.constant import *
import torch
from torch.autograd import Variable


class LstmMemory(BaseMemory):
    def __init__(self, memory_config):
        super(LstmMemory, self).__init__(memory_config)
        self.internal_memory = nn.LSTM(self.memory_config[OUTPUT_DIM], self.memory_config[OUTPUT_DIM], 1)
        self.hidden_state = None
        self.cell_state = None
        self.reset()


    def reset(self):
        self.hidden_state = Variable(torch.from_numpy(np.zeros(shape=(1, 1, self.memory_config[OUTPUT_DIM]))
                                                      )).float()
        self.cell_state = Variable(torch.from_numpy(np.zeros(shape=(1, 1, self.memory_config[OUTPUT_DIM]))
                                                      )).float()

    def get_summary_network(self):
        def _summariser():
            return self.hidden_state
        return _summariser

    def update_memory(self, history):
        # print(history.unsqueeze(0).size())
        # _, (self.hidden_state, _) = self.internal_memory(history, self.hidden_state)
        _, (self.hidden_state, self.cell_state) = self.internal_memory(history.unsqueeze(0), (self.hidden_state, self.cell_state))
        # self.internal_memory.append(Variable(torch.from_numpy(history)).float())
        # print(self.hidden_state.size())

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
        for param in self.internal_memory.named_parameters():
            if param[1].requires_grad:
                params.append(param[1])
        return params

    def _init_weights(self, module):
        if type(module) == nn.Linear:
            module.weight.data.fill_(1.0)

if __name__ == '__main__':
    memory = LstmMemory(memory_config=MemoryConfig())
    print(memory)
