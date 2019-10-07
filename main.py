import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class History(object):

    def __init__(self, size=1):
        self.size = size
        self._is_empty = True
        # will be set in _convert
        self._state = None

    def reset(self):
        self._is_empty = True

    def __call__(self, state):
        # only keep tiled representation
        state = state['tiles']
        converted = self.convert(state)
        return converted

    def convert(self, state):
        if self._is_empty:
            # add history dimension
            s = np.expand_dims(state, 0)
            # fill history with current state
            self._state = np.repeat(s, self.size, axis=0)
            self._is_empty = False
        else:
            # shift history
            self._state = np.roll(self._state, -1, axis=0)
            # add state to history
            self._state[-1] = state
        return np.concatenate(self._state, axis=0)


class Q(nn.Module):

    def __init__(self, nS, nA, device='cpu'):
        super(Q, self).__init__()

        self.nA = nA
        self.nS = nS
        self.device = device

        pad = 0
        nf = [nS[2], 32, 32]
        ks = [4, 2]
        ss = [2, 1]
        conv_h = nS[1]
        conv_w = nS[0]
        convs = []
        for i in range(len(ks)):
            conv = nn.Conv2d(nf[i], nf[i + 1], kernel_size=ks[i], stride=ss[i])
            convs.append(conv)
            conv_h = int((conv_h + 2 * pad - 1 * (ks[i] - 1) - 1) / ss[i] + 1)
            conv_w = int((conv_w + 2 * pad - 1 * (ks[i] - 1) - 1) / ss[i] + 1)
        self.convs = nn.ModuleList(convs)

        fc1_in = conv_h * conv_w * nf[-1]
        self.fc1 = nn.Linear(fc1_in, 64)
        self.fc2 = nn.Linear(64, nA)

    def forward(self, state, action):

        x = state.float()
        for conv in self.convs:
            x = F.relu(conv(x))
        b, c, h, w = x.shape

        x = self.fc1(x.view(b, c*w*h))
        x = F.relu(x)
        x = self.fc2(x)

        if action is not None:
            # mask must be byte or long tensor
            x = x[torch.arange(b, device=self.device),action.type(torch.int64)]
        return x


if __name__ == '__main__':
    from env.env import GameEnv
    from rl.dqn import DQN
    from rl.estimator import TargetEstimator
    import argparse

    parser = argparse.ArgumentParser(description='dqn_atari')
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--copy', default=1000, type=int)
    parser.add_argument('--mem-size', default=100000, type=int)
    parser.add_argument('--learn-start', default=1000, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--history-size', default=4, type=int)
    parser.add_argument('--episodes', default=1000, type=int)
    parser.add_argument('--load', default=None, type=str)

    args = parser.parse_args()
    print(args)

    logdir = f'runs/lr_{args.lr:.2E}/gamma_{args.gamma}/batch_{args.batch_size}/copy_{args.copy}/start_{args.learn_start}/'

    nA = 6
    device = 'cpu'
    env = GameEnv()

    if args.load is not None:
        q_est = TargetEstimator(args.load, lr=args.lr, copy_every=args.copy, device=device)
    else:
        q_model = Q((10, 10, 1 * args.history_size), nA, device=device).to(device)
        q_est = TargetEstimator(q_model, lr=args.lr, copy_every=args.copy, device=device)

    obs = History(size=args.history_size)

    agent = DQN(env,
                observe=obs,
                estimate_q=q_est,
                memory_size=args.mem_size,
                batch_size=args.batch_size,
                learn_start=args.learn_start,
                epsilon=0.15,
                gamma=args.gamma)

    # agent.eval()
    agent.train(args.episodes, max_steps=1000, logdir=logdir)