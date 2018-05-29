import os
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotter.util import metrics_key_set, matplotlib_args, agents, get_env_list, matplotlib_kwargs, compute_running_average
from utils.constant import *
from utils.log import parse_log_file
from utils.util import make_dir


def get_dir_to_save_plots(logs_path, dir_to_save_plots):
    dir_to_save_plots = (Path(dir_to_save_plots)
        .parent
        .joinpath(
        logs_path
            .as_posix()
            .rsplit("/", 1)[1]
    )).as_posix()

    make_dir(dir_to_save_plots)

    return dir_to_save_plots


def parse_logs_from_dir(logs_path, envs):
    dir_logs = {}
    for agent in agents:
        dir_logs[agent] = {}

    for log_idx, log_file_path in enumerate(logs_path.glob("**/log*.txt")):
        print("Parsing {}".format(log_file_path))
        for agent in agents:
            logs = parse_log_file(log_file_path=log_file_path, agent=agent, env_list=envs)
            for env in logs:
                if env not in dir_logs[agent]:
                    dir_logs[agent][env] = {}
                    for key in logs[env]:
                        if (key in metrics_key_set):
                            dir_logs[agent][env][key] = []
                for key in logs[env]:
                    if (key in metrics_key_set):
                        dir_logs[agent][env][key].append(logs[env][key])

    for agent in agents:
        for env in list(dir_logs[agent].keys()):
            for key in dir_logs[agent][env]:
                dir_logs[agent][env][key] = np.asarray(dir_logs[agent][env][key])

    return dir_logs


def transform_logs_to_aggregated_logs(dir_logs, window_size=1000):
    aggregated_logs = deepcopy(dir_logs)
    for agent, agent_val in dir_logs.items():
        for env, env_value in agent_val.items():
            for key, key_val in env_value.items():
                min_len = min(map(lambda x: len(x), key_val))
                metric_val = np.asarray(list(map(lambda x: compute_running_average(x[:min_len],
                                                                                   window_size=window_size), key_val)))
                _metric = {
                    AVERAGE: np.mean(metric_val, axis=0),
                    STDDEV: np.std(metric_val, axis=0)
                }
                aggregated_logs[agent][env][key] = _metric
    del dir_logs
    return aggregated_logs


def plot_from_dir(logs_path, env=MAZEBASE, dir_to_save_plots=None, last_n=0, window_size=1000):
    '''
    This method wraps the `
    :return:
    '''
    dir_to_save_plots = get_dir_to_save_plots(logs_path, dir_to_save_plots)

    envs = get_env_list(base_env=env)

    dir_logs = parse_logs_from_dir(logs_path=logs_path, envs=envs)

    np.save("{}/dir_logs.npy".format(dir_to_save_plots),
            dir_logs)

    aggregated_logs = transform_logs_to_aggregated_logs(dir_logs, window_size=window_size)

    for agent, agent_val in aggregated_logs.items():
        for env, env_value in agent_val.items():
            for key, key_val in env_value.items():
                if (last_n > 0):
                    plot_last_n_aggregated(agent_val, last_n, key, agent, dir_to_save_plots, env,
                                           *matplotlib_args,
                                           **matplotlib_kwargs)
                else:
                    plot_aggregated(agent_val, key, agent, dir_to_save_plots, env,
                                    *matplotlib_args,
                                    **matplotlib_kwargs)

    print(dir_to_save_plots)


def plot_aggregated(logs, key, agent, dir_to_save_plots, env, *args, **kwargs):
    new_metric = {
        AVERAGE: logs[env][key][AVERAGE],
        STDDEV: logs[env][key][STDDEV]
    }
    metric_average = new_metric[AVERAGE]
    std_average = new_metric[STDDEV]
    if (len(metric_average) > 2):
        plt.plot(metric_average, *args, **kwargs)
        # plt.show()
        ax = plt.gca()
        ax.fill_between(range(len(std_average)), metric_average + std_average, metric_average - std_average, alpha=0.2)
        # plt.show()
        if (key in set([CURRENT_EPISODIC_REWARD, AVERAGE_EPISODIC_REWARD])):
            ylabel = REWARD
            xlabel = "Number of Episodes"
            title = key
        elif (key in set([AVERAGE_BATCH_LOSS])):
            ylabel = LOSS
            xlabel = "Number of Batches"
            title = key
        elif (key in set([TIME])):
            ylabel = "time taken in self play"
            xlabel = "Number of Episodes"
            title = ylabel
        title = agent + "_" + title
        title = title + "___" + ENVIRONMENT + "_" + env
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.show()
        if dir_to_save_plots:
            path = os.path.join(dir_to_save_plots, title)
            plt.savefig(path)
        plt.clf()
    else:
        print("Not enough data to plot anything for key = {}, agent = {}, env = {}".format(key, agent, env))


def plot_last_n_aggregated(logs, n, key, agent, dir_to_save_plots, env, *args, **kwargs):
    recent_logs = {}
    recent_logs[env] = {}
    recent_logs[env][key] = logs[env][key]
    print("mean = {} for key = {}, agent = {}, env = {}".format(
        np.mean(np.asarray(recent_logs[env][key][AVERAGE])),
        key, agent, env))
    if (n > 0 and len(recent_logs[env][key]) > n):
        recent_logs[env][key] = recent_logs[env][key][-n:]
    if (key in [CURRENT_EPISODIC_REWARD, AVERAGE_EPISODIC_REWARD, AVERAGE_BATCH_LOSS, TIME]):
        return plot_aggregated(recent_logs, key, agent, dir_to_save_plots, env, *args, **kwargs)
