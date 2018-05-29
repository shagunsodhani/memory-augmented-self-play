from utils.constant import CURRENT_EPISODIC_REWARD, AVERAGE_EPISODIC_REWARD, AVERAGE_BATCH_LOSS, TIME, \
    ALICE, BOB, SELFPLAY
import numpy as np

matplotlib_args = ["--bo"]
matplotlib_kwargs = {"ms":1.0}
metrics_key_set = set([CURRENT_EPISODIC_REWARD, AVERAGE_EPISODIC_REWARD, AVERAGE_BATCH_LOSS, TIME])
agents = [ALICE, BOB]

def get_env_list(base_env):
    return [SELFPLAY + "_" + base_env, SELFPLAY + "_target_" + base_env, base_env, SELFPLAY + "_memory_" + base_env]

def compute_running_average(metric, window_size=1000):
    new_metric = []
    for i in range(window_size, len(metric)-window_size-1):
        new_metric.append(sum(metric[i-window_size:i])/window_size)
    return np.asarray(new_metric)
