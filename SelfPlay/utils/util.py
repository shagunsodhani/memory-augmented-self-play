import random

import numpy as np
import torch
import pathlib


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)