import os
import random
from time import time

import numpy as np
import torch

from utils.constant import *


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, data):
        pass

    def save_model(self, epochs=-1, optimisers=None, save_dir=None, name=ALICE, timestamp=None):
        '''
        Method to persist the model
        '''
        if not timestamp:
            timestamp = str(int(time()))
        state = {
            EPOCHS: epochs + 1,
            STATE_DICT: self.state_dict(),
            OPTIMISER: [optimiser.state_dict() for optimiser in optimisers],
            NP_RANDOM_STATE: np.random.get_state(),
            PYTHON_RANDOM_STATE: random.getstate(),
            PYTORCH_RANDOM_STATE: torch.get_rng_state()
        }
        path = os.path.join(save_dir,
                            name + "_model_timestamp_" + timestamp + ".tar")
        torch.save(state, path)
        print("saved model to path = {}".format(path))

    def load_model(self, optimisers, load_path=None, name=ALICE, timestamp=None):
        timestamp = str(timestamp)
        path = os.path.join(load_path,
                            name + "_model_timestamp_" + timestamp + ".tar")
        print("Loading model from path {}".format(path))
        checkpoint = torch.load(path)
        epochs = checkpoint[EPOCHS]
        self._load_metadata(checkpoint)
        self._load_model_params(checkpoint[STATE_DICT])

        for i, _ in enumerate(optimisers):
            optimisers[i].load_state_dict(checkpoint[OPTIMISER][i])
        return optimisers, epochs

    def _load_metadata(self, checkpoint):
        np.random.set_state(checkpoint[NP_RANDOM_STATE])
        random.setstate(checkpoint[PYTHON_RANDOM_STATE])
        torch.set_rng_state(checkpoint[PYTORCH_RANDOM_STATE])

    def _load_model_params(self, state_dict):
        self.load_state_dict(state_dict)
