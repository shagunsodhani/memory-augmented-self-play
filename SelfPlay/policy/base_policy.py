import os
import random
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.nn.init import xavier_uniform

from memory.base_memory import BaseMemory
from memory.lstm_memory import LstmMemory
from memory.memory_config import MemoryConfig
from utils.constant import *
from utils.log import write_loss_log


class BasePolicy(torch.nn.Module):
    def __init__(self, policy_config):
        super(BasePolicy, self).__init__()
        self.logits = []
        # This corresponds to the log(pi) values
        self.returns = []
        self.use_baseline = policy_config[USE_BASELINE]
        self.losses = [0.0, 0.0]
        self.update_frequency = int(policy_config[BATCH_SIZE])
        self.update_counter = 0
        self.shared_features = None
        self.actor = None
        self.critic = None
        self.is_self_play = policy_config[IS_SELF_PLAY]
        self.is_self_play_with_memory = bool(policy_config[IS_SELF_PLAY_WITH_MEMORY] * self.is_self_play)
        self.input_size = policy_config[INPUT_SIZE]
        self.shared_features_size = policy_config[SHARED_FEATURES_SIZE]
        self.shared_features_size_output = self.shared_features_size
        memory_config = MemoryConfig(episode_memory_size=policy_config[EPISODE_MEMORY_SIZE],
                                     input_dim=self.shared_features_size, output_dim=self.shared_features_size)
        if policy_config[MEMORY_TYPE] == BASE_MEMORY:
            self.memory = BaseMemory(memory_config=memory_config)
        elif policy_config[MEMORY_TYPE] == LSTM_MEMORY:
            self.memory = LstmMemory(memory_config=memory_config)

        if (self.is_self_play):
            self.input_size = self.input_size * 2
            if(self.is_self_play_with_memory):
                self.shared_features_size_output = self.shared_features_size*2
            # if (self.is_self_play_with_memory):
            #     self.input_size = self.input_size * 3
            # else:
            #     self.input_size = self.input_size * 2
        # Mote that the name of the environment and the agent are provided only for the sake of book-keeping
        self.bookkeeping = {}
        self.bookkeeping[ENVIRONMENT] = policy_config[ENVIRONMENT]
        if (self.is_self_play):
            self.bookkeeping[ENVIRONMENT] = SELFPLAY + "_" + self.bookkeeping[ENVIRONMENT]
        self.bookkeeping[AGENT] = policy_config[AGENT]
        self.num_actions = policy_config[NUM_ACTIONS]

        self.shared_features = nn.Sequential(
            nn.Linear(self.input_size, self.shared_features_size),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(self.shared_features_size_output, self.num_actions)
        )

    def update_memory(self, history):
        # Tuple of Observations(start_state, end_state)
        history = self.shared_features(Variable(torch.from_numpy(history)).float())
        self.memory.update_memory(history=history)

    def summarize_memory(self):
        return self.memory.summarize()

    def forward(self, data):
        pass

    def _to_do_update(self, agent_name):
        if (self.update_counter % self.update_frequency == 0):
            write_loss_log(average_batch_loss=(sum(self.losses)).data[0] / self.update_frequency, agent=agent_name,
                           environment=self.bookkeeping[ENVIRONMENT])
            return True
        return False

    def init_weights(self):
        self.init_weights_shared_features()
        self.init_weights_actor()
        self.init_weights_critic()

    def init_weights_shared_features(self):
        if self.shared_features:
            for layer in self.shared_features:
                self._init_weights_layer(layer)

    def init_weights_actor(self):
        if self.actor:
            for layer in self.actor:
                self._init_weights_layer(layer)

    def init_weights_critic(self):
        if self.critic:
            for layer in self.critic:
                self._init_weights_layer(layer)

    def _init_weights_layer(self, layer):
        '''Method to initialise the weights for a given layer'''
        if isinstance(layer, nn.Linear):
            xavier_uniform(layer.weight.data)
            # xavier_uniform(layer.bias.data)

    def save_model(self, epochs=-1, optimisers=None, save_dir=None, name=ALICE, timestamp=None):
        '''
        Method to persist the model
        '''
        if not timestamp:
            timestamp = str(int(time()))
        state = {
            EPOCHS: epochs + 1,
            STATE_DICT: self.state_dict(),
            OPTIMISER: [optimiser.state_dict() for optimiser in optimisers],
            NP_RANDOM_STATE: np.random.get_state(),
            PYTHON_RANDOM_STATE: random.getstate(),
            PYTORCH_RANDOM_STATE: torch.get_rng_state()
        }
        path = os.path.join(save_dir,
                            name + "_model_timestamp_" + timestamp + ".tar")
        torch.save(state, path)
        print("saved model to path = {}".format(path))

    def load_model(self, optimisers, load_path=None, name=ALICE, timestamp=None):
        timestamp = str(timestamp)
        path = os.path.join(load_path,
                            name + "_model_timestamp_" + timestamp + ".tar")
        print("Loading model from path {}".format(path))
        checkpoint = torch.load(path)
        epochs = checkpoint[EPOCHS]
        self._load_metadata(checkpoint)
        self._load_model_params(checkpoint[STATE_DICT])

        for i, _ in enumerate(optimisers):
            optimisers[i].load_state_dict(checkpoint[OPTIMISER][i])
        return optimisers, epochs

    def _load_metadata(self, checkpoint):
        np.random.set_state(checkpoint[NP_RANDOM_STATE])
        random.setstate(checkpoint[PYTHON_RANDOM_STATE])
        torch.set_rng_state(checkpoint[PYTORCH_RANDOM_STATE])

    def _load_model_params(self, state_dict):
        self.load_state_dict(state_dict)

    def get_reward(self, observation):
        if (self.is_self_play and isinstance(observation, tuple) and len(observation) > 1):
            # This should be modified in the future
            reward = observation[0].reward
        else:
            reward = observation.reward
        return reward

    def get_state(self, observation):
        if (self.is_self_play and isinstance(observation, tuple)):
            # Should be removed
            if (self.is_self_play_with_memory and len(observation) == 3):
                S = Variable(torch.cat((torch.from_numpy(observation[0].state).float(),
                                        torch.from_numpy(observation[1].state).float(),
                                        torch.from_numpy(observation[2].state).float())).unsqueeze(0))
            # elif (not self.is_self_play_with_memory and len(observation) == 2):
            elif (len(observation) == 2):
                S = Variable(torch.cat((torch.from_numpy(observation[0].state).float(),
                                        torch.from_numpy(observation[1].state).float())).unsqueeze(0))
        else:
            S = Variable((torch.from_numpy(observation.state).float().unsqueeze(0)))
        return S


class BasePolicyReinforce(BasePolicy):
    # def __init__(self, input_size=-1, batch_size=32, is_self_play=False):
    def __init__(self, policy_config):
        super(BasePolicyReinforce, self).__init__(policy_config)

        # super(BasePolicyReinforce, self).__init__(use_baseline=False, input_size=input_size, batch_size=batch_size, is_self_play=is_self_play)
        self.logits = []
        # This corresponds to the log(pi) values
        self.returns = []

    def forward(self, data):
        shared_features = F.relu(self.shared_features(data))
        if(self.is_self_play and self.is_self_play_with_memory):
            shared_features = torch.cat((F.relu(self.shared_features(data)),
                                     self.summarize_memory().unsqueeze(0).detach()), dim=1)
        action_logits = self.actor(shared_features)
        return F.softmax(action_logits, dim=1)

    def get_action(self, observation):
        reward = self.get_reward(observation)
        S = self.get_state(observation)
        self.returns.append(reward)
        action_prob = self.forward(S)
        distribution = Categorical(action_prob)
        action = distribution.sample()
        self.logits.append(distribution.log_prob(action))
        return action.data[0]

    def update(self, optimisers, gamma, agent_name):
        # In this case, the list of optimisers has just 1 value
        optimiser = optimisers[0]
        running_return = 0
        policy_loss = []
        _returns = []
        gamma_exps = []
        current_gamma_exp = 1.0
        for _return in self.returns[::-1]:
            running_return = _return + gamma * running_return
            _returns.insert(0, running_return)
            gamma_exps.append(current_gamma_exp)
            current_gamma_exp = current_gamma_exp * gamma
        _returns = torch.FloatTensor(_returns)
        gamma_exps = torch.FloatTensor(gamma_exps)
        _returns = (_returns - _returns.mean()) / (_returns.std(unbiased=False) + np.finfo(np.float32).eps)
        for logit, _return, current_gamma_exp in zip(self.logits, _returns, gamma_exps):
            policy_loss.append(-logit * _return * current_gamma_exp)
        total_loss = torch.cat(policy_loss).sum()
        self.losses[0] += total_loss

        self.update_counter += 1

        if (self._to_do_update(agent_name=agent_name)):
            optimiser.zero_grad()
            loss = self.losses[0] / self.update_frequency
            loss.backward()
            optimiser.step()
            self.losses[0] = 0.0
        self.returns = []
        self.logits = []
        return (optimiser,)


class BasePolicyReinforceWithBaseline(BasePolicy):
    def __init__(self, policy_config):
        super(BasePolicyReinforceWithBaseline, self).__init__(policy_config)
        self.logits = []
        # This corresponds to the log(pi) values
        self.returns = []
        self.state_values = []
        self.actor_params_names = set([SHARED_FEATURES, ACTOR])
        self.critic_params_names = set([SHARED_FEATURES, CRITIC])
        self._lambda = policy_config[LAMBDA]
        self.is_self_play = policy_config[IS_SELF_PLAY]
        self.critic = nn.Sequential(
            nn.Linear(self.shared_features_size_output, 1)
        )

    def forward(self, data):
        shared_features = F.relu(self.shared_features(data))
        if(self.is_self_play and self.is_self_play_with_memory):
            shared_features = torch.cat((shared_features,
                                     self.summarize_memory().unsqueeze(0).detach()), dim=1)
        action_logits = self.actor(shared_features)
        state_values = self.critic(shared_features)
        return F.softmax(action_logits, dim=1), state_values

    def get_action(self, observation):
        reward = self.get_reward(observation)
        S = self.get_state(observation)
        self.returns.append(reward)
        action_prob, state_value = self.forward(S)
        distribution = Categorical(action_prob)
        action = distribution.sample()
        self.logits.append(distribution.log_prob(action))
        self.state_values.append(state_value)
        return action.data[0]

    def get_actor_params(self):
        params = []
        for param in self.named_parameters():
            if param[0].split(".")[0] in self.actor_params_names and param[1].requires_grad:
                params.append(param[1])
        return params

    def get_memory_params(self):
        return self.memory.get_params()

    def get_critic_params(self):
        params = []
        for param in self.named_parameters():
            if param[0].split(".")[0] in self.critic_params_names and param[1].requires_grad:
                params.append(param[1])
        return params

    def update(self, optimisers, gamma, agent_name):
        num_optimisers = len(optimisers)
        if (num_optimisers == 1):
            optimiser = optimisers[0]
        elif (num_optimisers == 2):
            actor_optimiser, critic_optimiser = optimisers

        running_return = 0
        policy_loss = []
        state_value_loss = []
        _returns = []
        gamma_exps = []
        current_gamma_exp = 1.0
        for _return in self.returns[::-1]:
            running_return = _return + gamma * running_return
            _returns.insert(0, running_return)
            gamma_exps.append(current_gamma_exp)
            current_gamma_exp = current_gamma_exp * gamma
        _returns = torch.FloatTensor(_returns)
        gamma_exps = torch.FloatTensor(gamma_exps)
        _returns = (_returns - _returns.mean()) / (_returns.std(unbiased=False) + np.finfo(np.float32).eps)
        for logit, state_value, _return, current_gamma_exp in zip(self.logits, self.state_values, _returns, gamma_exps):
            state_value_loss.append(F.smooth_l1_loss(state_value, Variable(torch.Tensor([_return]))))
            _return = _return - state_value.data[0][0]
            policy_loss.append(-logit * _return * current_gamma_exp)

        self.returns = []
        self.logits = []
        self.state_values = []

        actor_loss = torch.cat(policy_loss).sum()
        critic_loss = torch.cat(state_value_loss).sum()

        self.update_counter += 1

        if (num_optimisers == 1):
            loss = actor_loss + self._lambda * critic_loss
            self.losses[0] += loss
            if (self._to_do_update(agent_name=agent_name)):
                optimiser.zero_grad()
                loss = self.losses[0] / self.update_frequency
                loss.backward()
                optimiser.step()
                self.losses[0] = 0.0
            return (optimiser,)

        elif (num_optimisers == 2):
            self.losses[0] += actor_loss
            self.losses[1] += critic_loss
            if (self._to_do_update(agent_name=agent_name)):
                actor_optimiser.zero_grad()
                actor_loss.backward(retain_graph=True)
                actor_optimiser.step()

                critic_optimiser.zero_grad()
                critic_loss.backward()
                critic_optimiser.step()

                self.losses[0] = 0.0
                self.losses[1] = 0.0

            return (actor_optimiser, critic_optimiser)
