from policy.base_policy import BasePolicyReinforce, BasePolicyReinforceWithBaseline
from utils.constant import *

default_input_size = 6


class AcrobotPolicyReinforce(BasePolicyReinforce):
    def __init__(self, policy_config):
        if not policy_config[INPUT_SIZE]:
            policy_config[INPUT_SIZE] = default_input_size
        policy_config[SHARED_FEATURES_SIZE] = 128
        super(AcrobotPolicyReinforce, self).__init__(policy_config)
        self.init_weights()


class AcrobotPolicyReinforceWithBaseline(BasePolicyReinforceWithBaseline):
    def __init__(self, policy_config):
        if not policy_config[INPUT_SIZE]:
            policy_config[INPUT_SIZE] = default_input_size
        policy_config[SHARED_FEATURES_SIZE] = 128
        super(AcrobotPolicyReinforceWithBaseline, self).__init__(policy_config)
        self.init_weights()
