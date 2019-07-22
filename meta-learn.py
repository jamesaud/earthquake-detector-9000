#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, linspace, nn, randperm, sin
from torch.autograd import Variable
from torch.nn import Linear
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from reptile.utils import ParamDict
from models import mnist_three_component_exp
from main import create_dataset, create_loader
from config import options
from utils import dotdict
import glob 
import os 
import copy
pj = os.path.join

configuration = 'meta_learning'
_settings = options[configuration]
_new_settings = lambda: dotdict(copy.deepcopy(_settings))
settings = _new_settings()


MODEL = mnist_three_component_exp

Weights = ParamDict
criterion = F.l1_loss


PLOT = True
INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE = 1, 64, 1
N = 50  # Use 50 evenly spaced points on sine wave.

LR, META_LR = 0.02, 0.1  # Copy OpenAI's hyperparameters.
BATCH_SIZE, META_BATCH_SIZE = 10, 3
EPOCHS, META_EPOCHS = 1, 30000
TEST_GRAD_STEPS = 2 ** 3
PLOT_EVERY = 3000

class WeightsModel(nn.Module):
    def __init__(self, weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if weights is not None:
            self.load_state_dict(deepcopy(weights))


class Model(MODEL, WeightsModel):
    pass



def _gen_task(num_pts=N) -> DataLoader:
    # amplitude
    a = np.random.uniform(low=0.1, high=5)  # amplitude
    b = np.random.uniform(low=0, high=2 * np.pi)  # phase

    # Need to make x N,1 instead of N, to avoid
    # https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float
    x = linspace(start=-5, end=5, steps=num_pts, dtype=torch.float)[:, None]
    y = a * sin(x + b)

    x.cuda()
    y.cuda()
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    return loader




def train_batch(x: Tensor, y: Tensor, model: Model, opt) -> None:
    """Statefully train model on single batch."""
    # x, y = cuda(Variable(x)), cuda(Variable(y))

    # TODO figure out why ray breaks if I just declare criterion at the top.
    # loss = F.mse_loss(model(x), y)
    loss = criterion(model(x), y)

    opt.zero_grad()
    loss.backward()
    opt.step()


def evaluate(model: Model, task: DataLoader, criterion=criterion) -> float:
    """Evaluate model on all the task data at once."""
    model.eval()

    with torch.no_grad():
        x, y = task.dataset.tensors
        loss = criterion(model(x), y)
        return float(loss)


def sgd(meta_weights: Weights, epochs: int) -> Weights:
    """Run SGD on a randomly generated task."""

    model = Model(meta_weights).cuda()
    model.train()  # Ensure model is in train mode.

    task = gen_task()
    opt = SGD(model.parameters(), lr=LR)

    for _ in range(epochs):
        for x, y in task:
            train_batch(x, y, model, opt)

    return model.state_dict()


def REPTILE(
    meta_weights: Weights, meta_batch_size: int = META_BATCH_SIZE, epochs: int = EPOCHS
) -> Weights:
    """Run one iteration of REPTILE."""
    weights = [sgd(meta_weights, epochs) for _ in range(meta_batch_size)]

    weights = [ParamDict(w) for w in weights]

    # TODO Implement custom optimizer that makes this work with builtin
    # optimizers easily. The multiplication by 0 is to get a ParamDict of the
    # right size as the identity element for summation.
    meta_weights += (META_LR / epochs) * sum(
        (w - meta_weights for w in weights), 0 * meta_weights
    )
    return meta_weights


def station_loader(station_path: str) -> DataLoader:
    station_settings = _new_settings()
    station_settings.train.path = station_path
    dataset = create_dataset(station_settings, Model.transformations['train'], train=True)
    train_loader = create_loader(dataset, train=True, weigh_classes = station_settings.weigh_classes) 


if __name__ == "__main__":
    import copy

    num_to_use = 5
    station_paths = glob.glob(pj(settings.train.path, '*'))[:num_to_use]
    loaders = [station_loader(path) for path in station_paths]
    
    


    # # Need to put model on GPU first for tensors to have the right type.
    # meta_weights = Model().to(device).state_dict()
    # # meta_weights = meta_weights_.state_dict()


    # for iteration in range(1, META_EPOCHS + 1):

    #     meta_weights = REPTILE(ParamDict(meta_weights))

    #     if iteration == 1 or iteration % PLOT_EVERY == 0:

    #         model = Model(meta_weights).to(device)
    #         model.train()  # set train mode
    #         opt = SGD(model.parameters(), lr=LR)

    #         for _ in range(TEST_GRAD_STEPS):
    #             train_batch(x_plot, y_plot, model, opt)

    #         print(f"Iteration: {iteration}\tLoss: {evaluate(model, plot_task):.3f}")
