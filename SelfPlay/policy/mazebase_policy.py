import torch.nn as nn

from policy.base_policy import BasePolicyReinforce, BasePolicyReinforceWithBaseline
from utils.constant import *

default_input_size = 10 * 10 * 156

class MazebasePolicyReinforce(BasePolicyReinforce):
    def __init__(self, policy_config):
        if not policy_config[INPUT_SIZE]:
            policy_config[INPUT_SIZE] = default_input_size
        policy_config[SHARED_FEATURES_SIZE] = 50
        super(MazebasePolicyReinforce, self).__init__(policy_config)
        self.init_weights()

class MazebasePolicyReinforceWithBaseline(BasePolicyReinforceWithBaseline):
    def __init__(self,policy_config):
        if not policy_config[INPUT_SIZE]:
            policy_config[INPUT_SIZE] = default_input_size
        policy_config[SHARED_FEATURES_SIZE] = 50
        super(MazebasePolicyReinforceWithBaseline, self).__init__(policy_config)
        self.shared_features = nn.Sequential(
            nn.Linear(self.input_size, self.shared_features_size),
            nn.Tanh()
        )
        self.init_weights()