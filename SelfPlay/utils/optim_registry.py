import torch.optim
from utils.constant import *


def get_supported_optimisers():
    return set([ADAM, SGD, RMSPROP])


def choose_optimiser(optimiser_name=ADAM):
    if (optimiser_name == ADAM):
        return torch.optim.Adam
    elif (optimiser_name == SGD):
        return torch.optim.SGD
    elif (optimiser_name == RMSPROP):
        return torch.optim.RMSprop
