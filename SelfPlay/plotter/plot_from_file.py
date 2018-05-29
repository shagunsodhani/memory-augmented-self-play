import os

import matplotlib.pyplot as plt
import numpy as np

from plotter.util import metrics_key_set, matplotlib_args, agents, get_env_list, matplotlib_kwargs, compute_running_average
from utils.constant import *
from utils.log import parse_log_file


def plot_from_file(log_file_path, env=MAZEBASE, dir_to_save_plots=None, last_n=0, window_size = 1000):
    envs = get_env_list(base_env=env)
    for agent in agents:
        logs = parse_log_file(log_file_path=log_file_path, agent=agent, env_list=envs)
        for key in metrics_key_set:
            for env in envs:
                if (env in logs):
                    if (last_n > 0):
                        plot_last_n(logs, last_n, key, agent, dir_to_save_plots, env, window_size, *matplotlib_args, **matplotlib_kwargs)
                    else:
                        plot(logs, key, agent, dir_to_save_plots, env, window_size, *matplotlib_args, **matplotlib_kwargs)


def plot(logs, key, agent, dir_to_save_plots, env, window_size, *args, **kwargs):
    new_metric = compute_running_average(logs[env][key], window_size=window_size)
    if (len(new_metric) > 2):
        plt.plot(new_metric, *args, **kwargs)
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


def plot_last_n(logs, n, key, agent, dir_to_save_plots, env, window_size, *args, **kwargs):
    recent_logs = {}
    recent_logs[env] = {}
    recent_logs[env][key] = logs[env][key]
    print("mean = {} for key = {}, agent = {}, env = {}".format(
        np.mean(np.asarray(recent_logs[env][key])),
        key, agent, env))
    if (n > 0 and len(recent_logs[env][key]) > n):
        recent_logs[env][key] = recent_logs[env][key][-n:]
    if (key in [CURRENT_EPISODIC_REWARD, AVERAGE_EPISODIC_REWARD, AVERAGE_BATCH_LOSS, TIME]):
        return plot(recent_logs, key, agent, dir_to_save_plots, env, window_size, *args, **kwargs)
