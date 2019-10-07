import numpy as np
import datetime
from dataclasses import dataclass, asdict
from collections import Iterable
import tensorboardX as tb
import cv2


@dataclass
class Log(object):

    total_steps: int
    episode: int
    episode_step: int
    reward: float


@dataclass
class Transition(object):
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminal: bool


class Agent(object):

    def record_frame(self, episode, record_every, frames=None):
        if not episode % record_every:
            frame = self.env.render('rgb_array')
            w, h = frame.shape[:2]
            r = h/w
            w = np.minimum(w, 100)
            h = int(r*w)
            frame = cv2.resize(frame, (h, w))
            frame = np.expand_dims(frame, axis=0)
            if frames is None:
                return frame
            else:
                return np.append(frames, frame, axis=0)

    def train(self, episodes, max_steps=float('inf'), logdir='runs/', record_every=50):
        self.writer = tb.SummaryWriter(logdir=logdir + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        total_steps = 0
        for e_i in range(1, episodes+1):
            e_step = 0
            e_reward = 0
            res = self.start(log=Log(total_steps, e_i, e_step, e_reward))
            frames = self.record_frame(e_i, record_every)
            while not res['terminal'] and e_step < max_steps:
                res = self.step(res, log=Log(total_steps, e_i, e_step, e_reward))
                frames = self.record_frame(e_i, record_every, frames)

                e_step += 1
                total_steps += 1
                e_reward += res['reward']
            log = Log(total_steps, e_i, e_step, e_reward)
            self.end(log=log, writer=self.writer)
            print(log)
            for k, v in asdict(log).items():
                if isinstance(v, Iterable):
                    for i, v_i in enumerate(v):
                        self.writer.add_scalar(k + '_{}'.format(i), v_i, e_i)
                else:
                    self.writer.add_scalar(k, v, e_i)
            if not e_i % record_every:
                frames = np.expand_dims(np.moveaxis(frames, -1, 1), 0)
                self.writer.add_video('episode_recording', frames, e_i, fps=24)

    def start(self, log=None):
        raise NotImplementedError()

    def step(self, params, log=None):
        raise NotImplementedError()

    def end(self, log=None, writer=None):
        pass