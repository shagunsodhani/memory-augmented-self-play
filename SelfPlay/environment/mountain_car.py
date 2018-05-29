import math
import numpy as np
from gym import spaces
from gym.utils import seeding
import random
from copy import deepcopy

from environment.env import Environment
from environment.observation import Observation
from utils.constant import MOUNTAINCAR


class MountainCar(Environment):
    def __init__(self):
        self.name = MOUNTAINCAR
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.state = None
        self.observation = None
        self.n_max_steps = 10000
        self.steps_elapsed = 0

        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.reset()

    def act(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action - 1) * 0.001 + math.cos(3 * position) * (-0.0025)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position == self.min_position and velocity < 0): velocity = 0

        reward = -1.0
        self.state = (position, velocity)
        self.steps_elapsed += 1

        self.observation = Observation(reward=reward,
                                       state=np.array(self.state),
                                       is_episode_over=self.is_over())
        return self.observe()

    def observe(self):
        return self.observation

    def reset(self):
        self.state = np.array([np.random.uniform(low=-0.6, high=-0.4), 0])
        self.observation = Observation(
            reward=0.0,
            state=np.array(self.state),
            is_episode_over=self.is_over()
        )
        self.steps_elapsed = 0
        return self.observe()


    def is_over(self):
        if self.steps_elapsed >= self.n_max_steps:
            return True
        return bool(self.state[0] >= self.goal_position)

    def _height(self, xs):
        return np.sin(3 * xs) * .45 + .55

    def display(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.min_position) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth / 4, clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth / 4, clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position - self.min_position) * scale
            flagy1 = self._height(self.goal_position) * scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)])
            flag.set_color(.8, .8, 0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos - self.min_position) * scale, self._height(pos) * scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()

    def all_possible_actions(self):
        return list(range(self.action_space.n))

    def set_seed(self, seed):
        # @todo
        pass

    def are_states_equal(self, state_1, state_2):
        return (np.linalg.norm(state_1 - state_2) < 0.2)

    def create_copy(self):
        return deepcopy(self.state)

    def load_copy(self, env_copy):
        self.state = env_copy

if __name__ == "__main__":
    game = MountainCar()

    game.reset()
    game.display()

    actions = game.all_possible_actions()
    print(actions)
    for i in range(10000):
        print(i)
        print("==============")
        _action = random.choice(actions)
        print(_action)
        game.act(_action)
        game.display()
        if game.is_over():
            break

    game.close()


