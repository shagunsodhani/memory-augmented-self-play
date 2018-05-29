from policy.acrobot_policy import AcrobotPolicyReinforce, AcrobotPolicyReinforceWithBaseline
from policy.cartpole_policy import CartpolePolicyReinforce, CartpolePolicyReinforceWithBaseline
from policy.mazebase_policy import MazebasePolicyReinforce, MazebasePolicyReinforceWithBaseline
from policy.mountaincar_policy import MountainCarPolicyReinforce, MountainCarPolicyReinforceWithBaseline
from policy.policy_config import PolicyConfig
from utils.constant import *


def choose_policy(env_name=ENVIRONMENT,
                  agent_type=REINFORCE_AGENT,
                  use_baseline=True,
                  input_size=10 * 10 * 156,
                  num_actions=10,
                  batch_size=32,
                  is_self_play=False,
                  is_self_play_with_memory=False,
                  _lambda=0.3,
                  agent=AGENT,
                  episode_memory_size=10):
    policy_name = env_name + "_" + POLICY + "_" + agent_type.split("_")[0]

    policy_config = PolicyConfig(env_name=env_name,
                                 agent_type=agent_type,
                                 use_baseline=use_baseline,
                                 input_size=input_size,
                                 num_actions=num_actions,
                                 batch_size=batch_size,
                                 is_self_play=is_self_play,
                                 is_self_play_with_memory=is_self_play_with_memory,
                                 _lambda=_lambda,
                                 agent=agent,
                                 episode_memory_size=episode_memory_size
                                 )

    if (use_baseline):
        policy_name += "_with_baseline"

    if (policy_name == MAZEBASE_POLICY_REINFORCE):
        return MazebasePolicyReinforce(policy_config)
    if (policy_name == MAZEBASE_POLICY_REINFORCE_WITH_BASELINE):
        return MazebasePolicyReinforceWithBaseline(policy_config)
    elif (policy_name == ACROBOT_POLICY_REINFORCE):
        return AcrobotPolicyReinforce(policy_config)
    elif (policy_name == ACROBOT_POLICY_REINFORCE_WITH_BASELINE):
        return AcrobotPolicyReinforceWithBaseline(policy_config)
    elif (policy_name == CARTPOLE_POLICY_REINFORCE):
        return CartpolePolicyReinforce(policy_config)
    elif (policy_name == CARTPOLE_POLICY_REINFORCE_WITH_BASELINE):
        return CartpolePolicyReinforceWithBaseline(policy_config)
    elif (policy_name == MOUNTAINCAR_POLICY_REINFORCE):
        return MountainCarPolicyReinforce(policy_config)
    elif (policy_name == MOUNTAINCAR_POLICY_REINFORCE_WITH_BASELINE):
        return MountainCarPolicyReinforceWithBaseline(policy_config)
