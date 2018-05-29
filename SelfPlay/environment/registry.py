from environment.acrobot import Acrobot
from environment.cartpole import CartPole
from environment.mazebase_wrapper import MazebaseWrapper
from environment.mountain_car import MountainCar
from environment.selfplay import SelfPlay
from environment.selfplay_memory import SelfPlayMemory

from utils.constant import *


def choose_env(env=MAZEBASE):
    if (env == ACROBOT):
        return Acrobot
    elif (env == CARTPOLE):
        return CartPole
    elif (env == MAZEBASE):
        return MazebaseWrapper
    elif (env == MOUNTAINCAR):
        return MountainCar
    elif (env == SELFPLAY):
        return SelfPlay


def get_supported_envs():
    return set([ACROBOT, CARTPOLE, MAZEBASE, MOUNTAINCAR, SELFPLAY])


def choose_selfplay(config):
    if (config[MODEL][IS_SELF_PLAY]):
        if (config[MODEL][IS_SELF_PLAY_WITH_MEMORY]):
            return SelfPlayMemory
        else:
            return SelfPlay
