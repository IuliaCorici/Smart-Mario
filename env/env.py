from env.game import Game
import numpy as np


class GameEnv(object):

    def __init__(self, level='env/level.csv'):

        self.game = Game(level)
        self.repeat_frame_skip = 4

    def reset(self):
        self.game.reset()
        state = self.game.state()
        self.agent_coord = state['coord'][0] + self.game.camera_x
        return state

    def step(self, action):

        for _ in range(self.repeat_frame_skip):
            self.game.step(action)

        state = self.game.state()
        dead = state['dead']
        goal = state['goal']
        coord = state['coord']

        reward = -1 + (coord[0] + self.game.camera_x - self.agent_coord)/20 + 100*goal - 100*dead
        done = dead or goal
        self.agent_coord = coord[0] + self.game.camera_x
        return state, reward, done, {'goal': goal, 'dead': dead, 'distance': self.agent_coord}

    def render(self, mode='rgb_array'):
        pixels = self.game.render(mode)
        pixels = np.swapaxes(pixels, 0, 1)
        return pixels
