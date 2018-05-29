import json
import logging

from utils.constant import *
import numpy as np

log_types = [CONFIG, REWARD, LOSS, AGENT, ENVIRONMENT]


def _format_log(log):
    return json.dumps(log)


def write_log(log):
    '''This is the default method to write a log. It is assumed that the log has already been processed
     before feeding to this method'''
    print(log)
    logging.info(log)


def read_logs(log):
    '''This is the single point to read any log message from the file since all the log messages are persisted as jsons'''
    default_data = {
        TYPE: ""
    }
    try:
        data = json.loads(log)
        if (not isinstance(data, dict)):
            data = default_data
    except json.JSONDecodeError as e:
        data = default_data
    return data


def _format_custom_logs(keys=[], raw_log={}, _type=REWARD):
    log = {}
    if (keys):
        for key in keys:
            if key in raw_log:
                log[key] = raw_log[key]
    else:
        log = raw_log
    log[TYPE] = _type
    return _format_log(log)


delimiter = "\t"
quantifier = ":"


def format_reward_log(**kwargs):
    '''
    Method to return the formatted string about reward information to be pushed into the logs
    '''
    # message = EPISODE + quantifier + " {}" + delimiter + \
    #           CURRENT_EPISODIC_REWARD + quantifier + " {:5f}" + delimiter + \
    #           AVERAGE_EPISODIC_REWARD + quantifier + " {:5f}"
    # return message.format(
    #     episode_number, current_episodic_reward, average_episodic_reward)

    return json.dumps(kwargs)


def format_config_log(config):
    '''
        Method to return the formatted string about config information to be pushed into the logs
        '''
    return json.dumps(config)


def format_loss_log(**kwargs):
    return json.dumps(kwargs)


def write_config_log(config):
    config[TYPE] = CONFIG
    log = _format_log(config)
    write_log(log)


def write_reward_log(**kwargs):
    keys = [CURRENT_EPISODIC_REWARD, AVERAGE_EPISODIC_REWARD, AGENT, ENVIRONMENT]
    log = _format_custom_logs(keys=keys, raw_log=kwargs, _type=REWARD)
    write_log(log)

def write_time_log(**kwargs):
    keys = [TIME_ALICE, TIME_BOB, AGENT, ENVIRONMENT]
    log = _format_custom_logs(keys=keys, raw_log=kwargs, _type=TIME)
    write_log(log)

def write_position_log(**kwargs):
    keys = [ALICE_START_POSITION, ALICE_END_POSITION, BOB_START_POSITION, BOB_END_POSITION]
    log = _format_custom_logs(keys=keys, raw_log=kwargs, _type=POSITION)
    write_log(log)


def write_loss_log(**kwargs):
    keys = [AVERAGE_BATCH_LOSS, AGENT, ENVIRONMENT]
    # if (AGENT, ENV in kwargs):
    #     keys.append(AGENT)
    log = _format_custom_logs(keys=keys, raw_log=kwargs, _type=LOSS)
    write_log(log)


def pprint(config):
    print(json.dumps(config, indent=4))


def parse_log_file(log_file_path, agent=None, env_list=None):
    logs = {}
    agent_keys = [CURRENT_EPISODIC_REWARD, AVERAGE_EPISODIC_REWARD, AVERAGE_BATCH_LOSS,
                  ENVIRONMENT, TIME]
    common_keys = [CONFIG]
    keys = agent_keys + common_keys
    for env in env_list:
        logs[env] = {}
        for key in keys:
            logs[env][key] = []
    with open(log_file_path, "r") as f:
        for line in f:
            data = read_logs(log=line)
            _type = data[TYPE]
            if (_type in [REWARD, LOSS]):
                if (data[AGENT] == agent):
                    if(data[ENVIRONMENT] in logs):
                        for key in agent_keys:
                            if key in data:
                                logs[data[ENVIRONMENT]][key].append(data[key])
            elif(_type == TIME):
                data[ENVIRONMENT] = env_list[0]
                if(TIME_ALICE in data):
                    _agent = ALICE
                    key = TIME_ALICE
                elif(TIME_BOB in data):
                    _agent = BOB
                    key = TIME_BOB
                if(_agent == agent):
                    logs[data[ENVIRONMENT]][TIME].append(data[key])
            else:
                if _type == CONFIG:
                    key = CONFIG
                    for env in env_list:
                        logs[env][key].append(data)

    for env in list(logs.keys()):
        keys = set([CURRENT_EPISODIC_REWARD, AVERAGE_EPISODIC_REWARD, AVERAGE_BATCH_LOSS, TIME])
        keys_to_modify = [CONFIG, ENVIRONMENT]
        to_delete = True
        for key in logs[env]:
            if(key in keys and len(logs[env][key]) > 1):
                logs[env][key] = np.asarray(logs[env][key])
                to_delete = False
        if (to_delete):
            del logs[env]
        else:
            for key in keys_to_modify:
                logs[env][key] = logs[env][key][0]

    return logs

