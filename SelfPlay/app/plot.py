import matplotlib

matplotlib.use('agg')
import sys

from utils.config import get_config
from utils.constant import *
from plotter.plot_from_dir import plot_from_dir
from plotter.plot_from_file import plot_from_file
from pathlib import Path


def run(logs_path, env=MAZEBASE, dir_to_save_plots=None, last_n=0, window_size=1):
    logs_path = Path(logs_path).resolve()
    print(logs_path)

    if (logs_path.is_dir()):
        plot_from_dir(logs_path=logs_path, env=env, dir_to_save_plots=dir_to_save_plots, last_n=last_n, window_size=window_size)
    elif (logs_path.is_file()):
        plot_from_file(log_file_path=logs_path, env=env, dir_to_save_plots=dir_to_save_plots, last_n=last_n, window_size=window_size)


if __name__ == '__main__':
    logs_path = None
    config = None
    window_size = 5000
    if (len(sys.argv) == 2):
        logs_path = sys.argv[1]
        config = get_config(use_cmd_config=False)
    else:
        config = get_config(use_cmd_config=True)
        logs_path = config[LOG][FILE_PATH]
    # logs_path = "/Users/shagun/projects/self-play/SelfPlay/selfplay-with-memory-logs1"
    env = config[MODEL][ENV]
    last_n = 0
    dir_to_save_plots = config[PLOT][BASE_PATH]
    run(logs_path=logs_path,
        env=env,
        dir_to_save_plots=dir_to_save_plots,
        last_n=last_n,
        window_size=window_size)
