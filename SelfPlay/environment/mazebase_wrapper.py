import random
from copy import deepcopy

import mazebase
# These weird import statements are taken from https://github.com/facebook/MazeBase/blob/23454fe092ecf35a8aab4da4972f231c6458209b/py/example.py#L12
import mazebase.games as mazebase_games
import numpy as np
from mazebase.games import curriculum
from mazebase.games import featurizers

from environment.env import Environment
from environment.observation import Observation
from utils.constant import *


class MazebaseWrapper(Environment):
    """
    Wrapper class over maze base environment
    """

    def __init__(self):
        super(MazebaseWrapper, self).__init__()
        self.name = MAZEBASE
        try:
            # Reference: https://github.com/facebook/MazeBase/blob/3e505455cae6e4ec442541363ef701f084aa1a3b/py/mazebase/games/mazegame.py#L454
            small_size = (10, 10, 10, 10)
            lk = curriculum.CurriculumWrappedGame(
                mazebase_games.LightKey,
                curriculums={
                    'map_size': mazebase_games.curriculum.MapSizeCurriculum(
                        small_size,
                        small_size,
                        (10, 10, 10, 10)
                    )
                }
            )

            game = mazebase_games.MazeGame(
                games=[lk],
                featurizer=mazebase_games.featurizers.GridFeaturizer()
            )


        except mazebase.utils.mazeutils.MazeException as e:
            print(e)
        self.game = game
        self.actions = self.game.all_possible_actions()

    def observe(self):
        game_observation = self.game.observe()
        # Logic borrowed from:
        # https://github.com/facebook/MazeBase/blob/23454fe092ecf35a8aab4da4972f231c6458209b/py/example.py#L192
        obs, info = game_observation[OBSERVATION]
        featurizers.grid_one_hot(self.game, obs)
        obs = np.array(obs)
        featurizers.vocabify(self.game, info)
        info = np.array(obs)
        game_observation[OBSERVATION] = np.concatenate((obs, info), 2).flatten()
        is_episode_over = self.game.is_over()
        return Observation(id=game_observation[ID],
                           reward=game_observation[REWARD],
                           state=game_observation[OBSERVATION],
                           is_episode_over=is_episode_over)

    def reset(self):
        try:
            self.game.reset()
        except Exception as e:
            print(e)
        return self.observe()

    def display(self):
        return self.game.display()

    def is_over(self):
        return self.game.is_over()

    def act(self, action):
        self.game.act(action=action)
        return self.observe()

    def all_possible_actions(self):
        return self.actions

    def set_seed(self, seed):
        # Not needed here as we already set the numpy seed
        pass

    def create_copy(self):
        return deepcopy(self.game.game)

    def load_copy(self, env_copy):
        self.game.game = env_copy

    def are_states_equal(self, state_1, state_2):
        return np.array_equal(state_1, state_2)


if __name__ == "__main__":
    env = MazebaseWrapper()
    env.display()
    actions = env.all_possible_actions()
    print(actions)
    for i in range(10):
        print("==============")
        _action = random.choice(actions)
        print(_action)
        env.act(_action)
        env.display()
