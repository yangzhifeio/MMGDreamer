from termcolor import colored
import torch

from model.networks.vqvae_networks.network import VQVAE
from typing import *

import sys
import time

def load_vqvae(vq_conf, vq_ckpt, opt=None):
    assert type(vq_ckpt) == str

    # init vqvae for decoding shapes
    mparam = vq_conf.model.params
    n_embed = mparam.n_embed
    embed_dim = mparam.embed_dim
    ddconfig = mparam.ddconfig

    n_down = len(ddconfig.ch_mult) - 1

    vqvae = VQVAE(ddconfig, n_embed, embed_dim)
    
    map_fn = lambda storage, loc: storage
    state_dict = torch.load(vq_ckpt, map_location=map_fn)
    if 'vqvae' in state_dict:
        vqvae.load_state_dict(state_dict['vqvae'])
    else:
        vqvae.load_state_dict(state_dict)

    print(colored('[*] VQVAE: weight successfully load from: %s' % vq_ckpt, 'blue'))
    vqvae.requires_grad = False

    vqvae.to(opt.hyper.device)
    vqvae.eval()
    return vqvae

class AverageAggregator(object):
    def __init__(self):
        self._value = 0.
        self._count = 0

    @property
    def value(self):
        return self._value / self._count

    @value.setter
    def value(self, val: float):
        self._value += val
        self._count += 1

    def update(self, val: float, n=1):
        self._value += val
        self._count += n


class StatsLogger(object):
    __INSTANCE = None

    def __init__(self):
        if StatsLogger.__INSTANCE is not None:
            raise RuntimeError("StatsLogger should not be directly created")

        self._values = dict()
        self._loss = AverageAggregator()
        self._output_files = [sys.stdout]

    def add_output_file(self, f):
        self._output_files.append(f)

    @property
    def loss(self):
        return self._loss.value

    @loss.setter
    def loss(self, val: float):
        self._loss.value = val

    def update_loss(self, val: float, n=1):
        self._loss.update(val, n)

    def __getitem__(self, key: str):
        if key not in self._values:
            self._values[key] = AverageAggregator()
        return self._values[key]

    def clear(self):
        self._values.clear()
        self._loss = AverageAggregator()
        for f in self._output_files:
            if f.isatty():  # if the file stream is interactive
                print(file=f, flush=True)

    def print_progress(self, epoch: Union[int, str], iter: int, precision="{:.5f}"):
        fmt = "[{}] [epoch {:4d} iter {:3d}] | loss: " + precision
        msg = fmt.format(time.strftime("%Y-%m-%d %H:%M:%S"), epoch, iter, self._loss.value)
        for k,  v in self._values.items():
            msg += " | " + k + ": " + precision.format(v.value)
        for f in self._output_files:
            if f.isatty():  # if the file stream is interactive
                print(msg + "\b"*len(msg), end="", flush=True, file=f)
            else:
                print(msg, flush=True, file=f)

    @classmethod
    def instance(cls):
        if StatsLogger.__INSTANCE is None:
            StatsLogger.__INSTANCE = cls()
        return StatsLogger.__INSTANCE
