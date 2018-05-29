import logging

from utils.config import get_config
from utils.constant import *
from utils.util import set_seed


def bootstrap():
    config = get_config()
    set_seed(seed=config[GENERAL][SEED])
    log_file_name = config[LOG][FILE_PATH]
    print("Writing logs to file name: {}".format(log_file_name))
    logging.basicConfig(filename=log_file_name, format='%(message)s', filemode='w', level=logging.DEBUG)
    return config
