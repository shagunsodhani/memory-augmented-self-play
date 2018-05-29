from agent.registry import choose_agent
from app.util import bootstrap
from environment.registry import choose_env
from utils.constant import *
from utils.log import write_config_log, write_reward_log


def run_episode(env, agent, optimisers, total_episodic_rewards, i_episode, max_steps_per_episode=1000):
    current_episodic_reward = 0.0
    env.reset()
    agent.set_initial_state()
    for t in range(max_steps_per_episode):  # Don't infinite loop while learning
        observation = env.observe()
        action = agent.get_action(observation)
        observation = env.act(action)
        current_episodic_reward += observation.reward
        if observation.is_episode_over:
            break
    optimisers = agent.update_policy(optimisers, observation=observation)
    total_episodic_rewards += current_episodic_reward

    if i_episode % 1 == 0:
        write_reward_log(episode_number=i_episode, current_episodic_reward=current_episodic_reward,
                         average_episodic_reward=total_episodic_rewards / i_episode, agent=agent.name,
                         environment=env.name)
    return agent, optimisers, total_episodic_rewards


def run(config):
    write_config_log(config)
    env = choose_env(env=config[MODEL][ENV])()
    possible_actions = env.all_possible_actions()
    agent = config[MODEL][AGENT]

    agent = choose_agent(agent_type=agent) \
        (config=config, possible_actions=possible_actions)
    optimisers = agent.get_optimisers(optimiser_name=config[MODEL][OPTIMISER])
    total_episodic_rewards = 0.0
    for i_episode in range(1, config[MODEL][NUM_EPOCHS] + 1):
        agent, optimisers, total_episodic_rewards = run_episode(env, agent, optimisers,
                                                                total_episodic_rewards, i_episode,
                                                                max_steps_per_episode=config[MODEL][
                                                                    MAX_STEPS_PER_EPISODE])


if __name__ == '__main__':
    config = bootstrap()
    run(config)
