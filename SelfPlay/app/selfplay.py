from copy import deepcopy
from time import time

import numpy as np
from agent.registry import choose_agent
from app.util import bootstrap
from environment.registry import choose_env, choose_selfplay
from environment.selfplay_target import SelfPlayTarget
from utils.constant import *
from utils.log import write_config_log, write_reward_log, write_time_log


def run_target_episode(selfplay_target, bob, optimisers_bob, max_steps_per_episode):
    current_episodic_reward = 0.0
    selfplay_target.reset()
    bob.set_initial_state()
    for t in range(max_steps_per_episode):  # Don't infinite loop while learning
        observation = selfplay_target.bob_observe()
        action = bob.get_action(observation)
        observation = selfplay_target.act(action)
        current_episodic_reward += observation[0].reward
        if observation[0].is_episode_over:
            break
    optimisers_bob = bob.update_policy(optimisers_bob, observation=observation)

    return bob, optimisers_bob, current_episodic_reward


def run_target_epochs(selfplay_target, bob, optimisers_bob, batch_size, total_episodic_rewards, total_episodes,
                      max_steps_per_episode=1000):
    for i in range(batch_size):
        bob, optimisers_bob, current_episodic_reward = run_target_episode(selfplay_target, bob,
                                                                          optimisers_bob, max_steps_per_episode)

        total_episodes += 1
        total_episodic_rewards += current_episodic_reward
        for (player, total_reward, current_reward) in [(BOB, total_episodic_rewards, current_episodic_reward)]:
            if total_episodes % 1 == 0:
                write_reward_log(episode_number=str(total_episodes), agent=player,
                                 current_episodic_reward=current_reward,
                                 average_episodic_reward=total_reward / total_episodes,
                                 environment=selfplay_target.name)

    return bob, optimisers_bob, total_episodic_rewards, total_episodes


def run_selfplay_episode(selfplay, alice, bob, optimisers_alice, optimisers_bob, reward_scale=1.0, tMax=80):
    tA = 0
    selfplay.reset()
    selfplay.alice_start()
    while True:
        tA += 1
        observation = selfplay.alice_observe()
        action = alice.get_action(observation)
        observation = selfplay.act(action)
        if (tA >= tMax or observation[0].is_episode_over):
            selfplay.alice_stop()
            break
    write_time_log(time_alice=tA, agent=ALICE, environment=selfplay.name, time=tA)
    # write_position_log(alice_start_position=list(selfplay.alice_observations.start.state.astype(np.float64)),
    #                    alice_end_position=list(selfplay.alice_observations.end.state.astype(np.float64)))

    selfplay.bob_start()
    tB = 0
    while True:
        observation = selfplay.bob_observe()
        if (observation[0].is_episode_over or tA + tB >= tMax):
            if (observation[0].is_episode_over):
                print("solved")
            else:
                print("notsolved")
            selfplay.bob_stop()
            break
        tB += 1
        action = bob.get_action(observation)
        selfplay.act(action)
    write_time_log(time_bob=tB, agent=BOB, environment=selfplay.name, time=tA)
    # write_position_log(bob_end_position=list(observation[0].state.astype(np.float64)))

    rA = reward_scale * max(0, tB - tA)
    rB = -reward_scale * (tB)
    alice_reward_observation = selfplay.alice_observations.end
    alice_reward_observation.reward = rA
    alice.get_action(observation=(alice_reward_observation, alice_reward_observation))

    bob_reward_observation = selfplay.observe()
    bob_reward_observation.reward = rB
    bob.get_action(observation=(bob_reward_observation, bob_reward_observation))

    optimisers_alice = alice.update_policy(optimisers_alice)
    optimisers_bob = bob.update_policy(optimisers_bob)

    return alice, bob, optimisers_alice, optimisers_bob, rA, rB, selfplay


def run_selfplay_epoch(selfplay, alice, bob, optimisers_alice, optimisers_bob, batch_size,
                       total_rA, total_rB, total_episodes, reward_scale=1.0, tMax=80, use_memory=False):
    for i in range(batch_size):
        alice, bob, optimizers_alice, optimizers_bob, rA, rB, selfplay = run_selfplay_episode(selfplay, alice, bob,
                                                                                              optimisers_alice,
                                                                                              optimisers_bob,
                                                                                              reward_scale=reward_scale,
                                                                                              tMax=tMax)
        if (use_memory):
            alice_history = np.concatenate((selfplay.alice_observations.start.state.reshape(1, -1),
                                            selfplay.alice_observations.end.state.reshape(1, -1)), axis=1)

            alice.policy.update_memory(alice_history)

        total_rA += rA
        total_rB += rB
        total_episodes += 1

        for (player, total_reward, current_reward) in [(ALICE, total_rA, rA), (BOB, total_rB, rB)]:
            if total_episodes % 1 == 0:
                write_reward_log(episode_number=str(total_episodes), agent=player,
                                 current_episodic_reward=current_reward,
                                 average_episodic_reward=total_reward / total_episodes,
                                 environment=selfplay.name)

    return alice, bob, optimizers_alice, optimizers_bob, total_rA, total_rB, total_episodes


def run(config):
    config_alice = deepcopy(config)
    config_bob = deepcopy(config)
    config_bob[MODEL][IS_SELF_PLAY_WITH_MEMORY] = False
    write_config_log(config_alice)
    write_config_log(config_bob)

    use_memory = config[MODEL][IS_SELF_PLAY] and config[MODEL][IS_SELF_PLAY_WITH_MEMORY]
    task = config[MODEL][SELFPLAY_TYPE]
    env_for_selfplay = choose_env(env=config[MODEL][ENV])()
    env_for_selfplay_target = choose_env(env=config[MODEL][ENV])()
    agent = config[MODEL][AGENT]
    batch_size = config[MODEL][BATCH_SIZE]
    reward_scale = config[MODEL][REWARD_SCALE]
    max_steps_per_episode_self_play = config[MODEL][MAX_STEPS_PER_EPISODE_SELFPLAY]
    target_to_selfplay_ratio = config[MODEL][TARGET_TO_SELFPLAY_RATIO]

    #################Self Play Env#################
    selfplay = choose_selfplay(config=config_alice)(environment=env_for_selfplay, task=task)
    # selfplay = SelfPlayEnv(environment=env_for_selfplay, task=task)
    possible_actions_alice = selfplay.all_possible_actions(agent=ALICE)
    possible_actions_bob = selfplay.all_possible_actions(agent=BOB)

    alice = choose_agent(agent_type=agent) \
        (config=config, possible_actions=possible_actions_alice, name=ALICE)
    bob = choose_agent(agent_type=agent) \
        (config=config_bob, possible_actions=possible_actions_bob, name=BOB)

    # I am not sure why we want this
    selfplay.reset()

    optimisers_alice = alice.get_optimisers(optimiser_name=config[MODEL][OPTIMISER])
    optimisers_bob = bob.get_optimisers(optimiser_name=config[MODEL][OPTIMISER])
    total_rA = 0.0
    total_rB = 0.0

    if (config[MODEL][LOAD]):
        timestamp = config[MODEL][LOAD_TIMESTAMP]
        alice.load_model(optimizers=optimisers_alice, name=ALICE, timestamp=timestamp)
        bob.load_model(optimizers=optimisers_bob, name=BOB, timestamp=timestamp)

    #################Self Play Target Env#################
    selfplay_target = SelfPlayTarget(environment=env_for_selfplay_target)
    total_target_episodic_rewards = 0.0

    episode_counter_selfplay = 0.0
    episode_counter_target = 0.0

    for i_epoch in range(config[MODEL][NUM_EPOCHS]):

        alice, bob, optimizers_alice, optimizers_bob, total_rA, total_rB, \
        episode_counter_selfplay = run_selfplay_epoch(selfplay, alice, bob, optimisers_alice, optimisers_bob,
                                                      batch_size,
                                                      total_rA, total_rB, total_episodes=episode_counter_selfplay,
                                                      reward_scale=reward_scale, tMax=max_steps_per_episode_self_play,
                                                      use_memory=use_memory)
        bob, optimizers_bob, total_target_episodic_rewards, episode_counter_target = run_target_epochs(
            selfplay_target=selfplay_target,
            bob=bob, optimisers_bob=optimisers_bob, batch_size=batch_size * target_to_selfplay_ratio,
            total_episodic_rewards=total_target_episodic_rewards, total_episodes=episode_counter_target,
            max_steps_per_episode=config[MODEL][MAX_STEPS_PER_EPISODE]
        )
        if (i_epoch % config[MODEL][PERSIST_PER_EPOCH] == 0):
            timestamp = str(int(time() * 10000))
            alice.save_model(epochs=i_epoch, optimisers=optimisers_alice, name=ALICE, timestamp=timestamp)
            bob.save_model(epochs=i_epoch, optimisers=optimisers_bob, name=BOB, timestamp=timestamp)


if __name__ == '__main__':
    config = bootstrap()
    config[MODEL][IS_SELF_PLAY] = True
    run(config)
