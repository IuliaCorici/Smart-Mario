import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import inspect
import tarfile


def np_to_torch(data, device='cpu'):
    if type(data) == tuple:
        converted = tuple()
        for item in data:
            converted += (np_to_torch(item, device),)
    else:
        converted = torch.from_numpy(data.astype(np.float32)).to(device) if data is not None else None
    return converted


def torch_to_np(data):
    if type(data) == tuple:
        converted = tuple()
        for item in data:
            converted += (torch_to_np(item),)
    else:
        converted = data.detach().cpu().numpy() if data is not None else None
    return converted


class Estimator(object):

    def __init__(self, model, lr=1e-3, clip_grad_norm=None, clamp_loss=None, device='cpu'):
        self.device = device

        if type(model) == str:
            self.load(model)
        else:
            self.model = model
            self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.clip_grad_norm = clip_grad_norm
        self.clamp_loss = clamp_loss

    def __call__(self, *net_args):
        preds = self.model(*[np_to_torch(a, self.device) for a in net_args])
        return torch_to_np(preds)

    def update(self, targets, observations, actions):
        self.opt.zero_grad()

        observations = np_to_torch(observations, self.device)
        actions = np_to_torch(actions, self.device)
        targets = np_to_torch(targets, self.device)

        preds = self.model(observations, actions)
        l = F.mse_loss(preds.squeeze(), targets.float(), reduction='none')

        if self.clamp_loss is not None:
            l = torch.clamp(l, min=-self.clamp_loss, max=self.clamp_loss)
        l = l.mean()

        l.backward()
        if self.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

        self.opt.step()

        return float(l)

    def save(self, filename):
        checkpoint = {'model': self.model,
                      'opt': self.opt}
        torch.save(checkpoint, '/tmp/model.cp')
        with open('/tmp/definition.txt', 'w') as f:
            f.write(inspect.getsource(type(self.model)))
        with tarfile.open(filename, mode='w') as f:
            f.add('/tmp/model.cp')
            f.add('/tmp/definition.txt')

    def load(self, filename):
        with tarfile.open(filename, mode='r') as f:
            df = f.extractfile('tmp/definition.txt')
            d = df.read()
            exec(d)
            mf = f.extractfile('tmp/model.cp')
            checkpoint = torch.load(mf)
            self.model = checkpoint['model']
            self.opt = checkpoint['opt']


class TargetEstimator(Estimator):

    def __init__(self, model, lr=1e-3,tau=1., copy_every=0, **kwargs):
        super(TargetEstimator, self).__init__(model, lr=lr, **kwargs)

        self.target_model = copy.deepcopy(self.model)
        self.copy_every = copy_every
        self.tau = tau

    def should_copy(self, step):
        return self.copy_every and not step % self.copy_every

    def update_target(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1. - self.tau) * target_param.data)

    def __call__(self, *net_args, use_target_network=False):
        net = self.target_model if use_target_network else self.model
        preds = net(*[np_to_torch(a, self.device) for a in net_args])
        return torch_to_np(preds)