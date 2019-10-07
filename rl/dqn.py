import torch
from rl.agent import Agent, Transition
import numpy as np
from dataclasses import dataclass, astuple
from pathlib import Path
import random


@dataclass
class BatchTransition(object):
    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_observation: np.ndarray
    terminal: np.ndarray


class Memory(object):

    def __init__(self, size=100000):

        self.size = size
        self.memory = []
        self.current = 0

    def add(self, transition):
        if len(self.memory) < self.size:
            self.memory.append(None)
        self.memory[self.current] = np.array(astuple(transition))
        self.current = (self.current + 1) % self.size

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        batch = BatchTransition(*[np.array(i) for i in zip(*batch)])
        return batch


class DQN(Agent):

    def __init__(self, env,
                 observe=lambda x: x,
                 estimate_q=None,
                 memory_size=100000,
                 batch_size=1,
                 learn_start=0,
                 gamma=0.9,
                 epsilon=0.15
                 ):
        self.env = env
        self.observe = observe
        self.estimate_q = estimate_q
        self.memory = Memory(memory_size)

        self.gamma = gamma
        self.batch_size = batch_size
        self.learn_start = learn_start
        self.epsilon = epsilon

    def e_greedy(self, q_s):
        if np.random.rand() >= self.epsilon:
            return np.random.choice(np.argwhere(q_s == np.amax(q_s)).flatten())
        else:
            return np.random.choice(range(len(q_s)))
    # def softmax(self, q_s, log):
    #     exps = np.exp(np.array(q_s) * self.epsilon) # epsilon = inv_tau
    #     probas = exps / (np.sum(exps) + 1e-8)
    #     self.writer.add_scalar(f'prob_softmax', np.std(probas), log.total_steps)
    #     return np.random.choice(range(len(q_s)), p=probas)

    def eval(self):
        self.epsilon = 0
        self.learn_start = np.inf

    def start(self, log=None):
        state = self.env.reset()
        self.observe.reset()
        obs = self.observe(state)
        return {'observation': obs,
                'terminal': False}

    def step(self, previous, log=None):
        # get q estimates from current state, expand to make one-sized batch. returns one-sized batch, get that sample
        q_s = self.estimate_q(np.expand_dims(np.array(previous['observation']), 0), None)[0]
        # action = self.softmax(q_s, log)
        action = self.e_greedy(q_s)  
        next_state, reward, terminal, _ = self.env.step(action)
        next_obs = self.observe(next_state)

        # add in replay memory
        t = Transition(observation=previous['observation'],
                       action=action,
                       reward=reward,
                       next_observation=next_obs,
                       terminal=terminal)
        self.memory.add(t)
        if log.total_steps >= self.learn_start:
            batch = self.memory.sample(self.batch_size)

            # compute targets
            q_ns = self.estimate_q(batch.next_observation, None, use_target_network=True)
            max_q_ns = self.gamma*np.max(q_ns, axis=1)
            # immediate reward if final step, else reward + discounted Q-value
            q_target = batch.reward + max_q_ns*np.logical_not(batch.terminal)

            e_loss = self.estimate_q.update(q_target,
                                            batch.observation,
                                            batch.action)

            if self.estimate_q.should_copy(log.total_steps):
                self.estimate_q.update_target()

            self.writer.add_scalar(f'q_loss', e_loss, log.total_steps)
        return {'observation': next_obs,
                'reward': reward,
                'terminal': terminal}

    def end(self, log=None, writer=None):

        if (log.episode + 1) % 200 == 0:
            f = Path(list(writer.all_writers.keys())[0]) / 'checkpoints' / 'q_est_{}.tar'.format(log.episode)
            f.parents[0].mkdir(parents=True, exist_ok=True)
            self.estimate_q.save(f)


