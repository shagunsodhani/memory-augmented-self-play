import torch.nn as nn

from policy.base_policy import BasePolicyReinforce, BasePolicyReinforceWithBaseline
from utils.constant import *

default_input_size = 4


class CartpolePolicyReinforce(BasePolicyReinforce):
    def __init__(self, policy_config):
        if not policy_config[INPUT_SIZE]:
            policy_config[INPUT_SIZE] = default_input_size
        policy_config[SHARED_FEATURES_SIZE] = 10
        super(CartpolePolicyReinforce, self).__init__(policy_config)
        self.init_weights()


class CartpolePolicyReinforceWithBaseline(BasePolicyReinforceWithBaseline):
    def __init__(self, policy_config):
        if not policy_config[INPUT_SIZE]:
            policy_config[INPUT_SIZE] = default_input_size
        policy_config[SHARED_FEATURES_SIZE] = 10
        super(CartpolePolicyReinforceWithBaseline, self).__init__(policy_config)
        self.init_weights()
