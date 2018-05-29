from agent.human_agent import HumanAgent
from agent.random_agent import RandomAgent
from agent.reinforce_agent import ReinforceAgent
from utils.constant import RANDOM, HUMAN, REINFORCE


def choose_agent(agent_type=RANDOM):
    if (agent_type == RANDOM):
        return RandomAgent
    elif (agent_type == HUMAN):
        return HumanAgent
    elif (agent_type == REINFORCE):
        return ReinforceAgent


def get_supported_agents():
    return set([RANDOM, HUMAN, REINFORCE])
