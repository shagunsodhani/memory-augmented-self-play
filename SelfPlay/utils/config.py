import os
from configparser import ConfigParser
from datetime import datetime

from agent.registry import get_supported_agents
from environment.registry import get_supported_envs
from utils.argument_parser import argument_parser
from utils.constant import *
from utils.optim_registry import get_supported_optimisers
from utils.util import make_dir


def _read_config():
    '''
    Method to read the config file and return as a dict
    :return:
    '''
    config = ConfigParser()
    path = os.path.abspath(os.pardir).split('/SelfPlay')[0]
    config.read(os.path.join(path, 'SelfPlay/config', 'config.cfg'))
    return config._sections


def _get_boolean_value(value):
    if (value.lower() == TRUE):
        return True
    else:
        return False


def get_config(use_cmd_config = True):
    '''Method to prepare the config for all downstream tasks'''

    # Read the config file
    config = _read_config()

    if(use_cmd_config):
        config = argument_parser(config)

    if (config[GENERAL][BASE_PATH] == ""):
        base_path = os.getcwd().split('/SelfPlay')[0]
        config[GENERAL][BASE_PATH] = base_path

    if (config[GENERAL][DEVICE] == ""):
        config[GENERAL][DEVICE] = CPU

    for key in [SEED]:
        config[GENERAL][key] = int(config[GENERAL][key])

    key = ID
    if config[GENERAL][key] == "":
        config[GENERAL][key] = str(config[GENERAL][SEED])

    # Model Params
    for key in [NUM_EPOCHS, BATCH_SIZE, PERSIST_PER_EPOCH, EARLY_STOPPING_PATIENCE,
                NUM_OPTIMIZERS, LOAD_TIMESTAMP, MAX_STEPS_PER_EPISODE, MAX_STEPS_PER_EPISODE_SELFPLAY,
                TARGET_TO_SELFPLAY_RATIO, EPISODE_MEMORY_SIZE]:
        config[MODEL][key] = int(config[MODEL][key])

    for key in [LEARNING_RATE, GAMMA, LAMBDA, LEARNING_RATE_ACTOR, LEARNING_RATE_CRITIC, REWARD_SCALE]:
        config[MODEL][key] = float(config[MODEL][key])

    for key in [USE_BASELINE, LOAD, IS_SELF_PLAY, IS_SELF_PLAY_WITH_MEMORY]:
        config[MODEL][key] = _get_boolean_value(config[MODEL][key])

    agent = config[MODEL][AGENT]

    if (agent not in get_supported_agents()):
        config[MODEL][AGENT] = REINFORCE

    env = config[MODEL][ENV]

    if (env not in get_supported_envs()):
        config[MODEL][ENV] = MAZEBASE

    optimiser = config[MODEL][OPTIMISER]
    if (optimiser not in get_supported_optimisers()):
        config[MODEL][OPTIMISER] = ADAM

    if (config[MODEL][SAVE_DIR] == ""):
        config[MODEL][SAVE_DIR] = os.path.join(config[GENERAL][BASE_PATH], "model")
    elif (config[MODEL][SAVE_DIR][0] != "/"):
        config[MODEL][SAVE_DIR] = os.path.join(config[GENERAL][BASE_PATH], config[MODEL][SAVE_DIR])

    make_dir(config[MODEL][SAVE_DIR])

    if (config[MODEL][LOAD_PATH] == ""):
        config[MODEL][LOAD_PATH] = os.path.join(config[GENERAL][BASE_PATH], "model")

    elif (config[MODEL][LOAD_PATH][0] != "/"):
        config[MODEL][LOAD_PATH] = os.path.join(config[GENERAL][BASE_PATH], config[MODEL][LOAD_PATH])

    # TB Params
    config[TB][DIR] = os.path.join(config[TB][BASE_PATH], datetime.now().strftime('%b%d_%H-%M-%S'))
    config[TB][SCALAR_PATH] = os.path.join(config[TB][BASE_PATH], "all_scalars.json")

    # Log Params
    key = FILE_PATH
    if (config[LOG][key] == ""):
        config[LOG][key] = os.path.join(config[GENERAL][BASE_PATH],
                                        "SelfPlay",
                                        "log_{}.txt".format(str(config[GENERAL][SEED])))

    # Plot Params
    if (config[PLOT][BASE_PATH] == ""):
        config[PLOT][BASE_PATH] = os.path.join(config[GENERAL][BASE_PATH], "plot", config[GENERAL][ID])

    make_dir(path=config[PLOT][BASE_PATH])

    return config
