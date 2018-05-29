from app.env_reinforce import run
from app.util import bootstrap
from utils.constant import *


def main():
    config = bootstrap()
    ##############
    config[MODEL][ENV] = MAZEBASE
    config[MODEL][AGENT] = REINFORCE
    config[MODEL][USE_BASELINE] = True
    ##############

    run(config=config)


if __name__ == '__main__':
    main()
