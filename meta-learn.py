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
from utils import dotdict, reduce_dataset
import glob 
import os 
import copy
from typing import List, Callable
from pytorch_utils.utils import evaluation
import sys

pj = os.path.join

configuration = 'meta_learning'

# Generate new settings when needed for creating different data loaders for each station... 
_settings = options[configuration]
_new_settings = lambda: dotdict(copy.deepcopy(_settings))
settings = _new_settings()


MODEL = mnist_three_component_exp

Weights = ParamDict
criterion = nn.CrossEntropyLoss().cuda() #F.l1_loss


PLOT = True
SAMPLES_PER_TASK = 50    # Samples for each station: 50

LR, META_LR = 0.02, 0.1  # Copy OpenAI's hyperparameters.
BATCH_SIZE, META_BATCH_SIZE = 10, 3
EPOCHS, META_EPOCHS = 1, 30000
TEST_EPOCHS = 2 ** 3
PLOT_EVERY = 10  # 3000

class WeightsModel(nn.Module):
    def __init__(self, weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if weights is not None:
            self.load_state_dict(deepcopy(weights))


class Model(WeightsModel, MODEL):
    pass


class TaskCreator:
    """
    Class to create tasks given a list of station paths.
    Tasks are dataloaders, with references kept in self.station_loaders
    """
    def __init__(self, station_paths: List):
        self.station_paths = station_paths
        self.station_loaders = self._station_loaders(station_paths)
        self.__call__index = 0

    def __call__(self) -> DataLoader:
        """
        Cycles through station loaders everytime it is called, like an infinite generator (but not implemented as __iter__)
        """
        i = self.__call__index
        self._next__call__index()
        return self.station_loaders[i]

    def _next__call__index(self):
        self.__call__index = len(self.station_loaders) % (self.__call__index + 1)

    @staticmethod
    def _station_loaders(station_paths):
        return [station_loader(path) for path in station_paths]
    

def train_batch(x: List[Tensor], y: Tensor, model: Model, opt) -> None:
    """
    Statefully train model on single batch.
    """
    x, y = [Variable(component).cuda() for component in x], Variable(y).cuda()

    # TODO figure out why ray breaks if I just declare criterion at the top.
    # loss = F.mse_loss(model(x), y)
    x_predicted = model(x)
    loss = criterion(x_predicted, y)

    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss


def evaluate(model: Model, task: DataLoader, criterion=criterion) -> float:
    """Evaluate model on all the task data at once."""
    model.eval()

    with torch.no_grad():
        x, y = task.dataset.tensors
        loss = criterion(model(x), y)
        return float(loss)

def train_epoch(task, model, opt):
    cum_loss = 0
    for i, (x, y) in enumerate(task):
        loss = train_batch(x, y, model, opt)
        cum_loss += float(loss.data)
    return cum_loss

def train_epochs(task, model, opt, epochs):
    loss_per_epoch = []
    for _ in range(epochs):
        loss = train_epoch(task, model, opt)
        loss_per_epoch.append(loss)
    return loss_per_epoch

def sgd(meta_weights: Weights, epochs: int, gen_task: Callable) -> Weights:
    """Run SGD on a randomly generated task."""
    model = Model(meta_weights).cuda()
    model.train()  # Ensure model is in train mode.

    task = gen_task()
    opt = SGD(model.parameters(), lr=LR)

    train_epochs(task, model, opt, epochs)

    return model.state_dict()


def REPTILE(meta_weights: Weights,
            gen_task: Callable,
            meta_batch_size: int = META_BATCH_SIZE, 
            epochs: int = EPOCHS) -> Weights:
            
    """Run one iteration of REPTILE."""
    weights = [sgd(meta_weights, epochs, gen_task) for _ in range(meta_batch_size)]

    weights = [ParamDict(w) for w in weights]

    # TODO Implement custom optimizer that makes this work with builtin
    # optimizers easily. The multiplication by 0 is to get a ParamDict of the
    # right size as the identity element for summation.
    meta_weights += (META_LR / epochs) * sum(
        (w - meta_weights for w in weights), 0 * meta_weights
    )
    return meta_weights


def station_loader(station_path: str, num_samples=SAMPLES_PER_TASK) -> DataLoader:
    station_settings = _new_settings()
    station_settings.train.path = station_path

    dataset = create_dataset(station_settings, Model.transformations['train'], train=True)
    if num_samples:
        reduce_dataset(dataset, num_samples, copy_dataset=False)

    train_loader = create_loader(dataset, train=True, weigh_classes=station_settings.weigh_classes, batch_size=BATCH_SIZE) 
    return train_loader


if __name__ == "__main__":
    #TODO: Test on the full dataset at the end after meta-learning 

    num_to_use = 5
    station_paths = glob.glob(pj(settings.train.path, '*'))
    test_path = station_paths.pop()
 
    gen_task = TaskCreator(station_paths[:num_to_use])    # Training tasks
    test_task = TaskCreator([test_path])()   # Evaluation task

    # Need to put model on GPU first for tensors to have the right type.
    meta_weights = Model().cuda().state_dict()

    for iteration in range(1, META_EPOCHS + 1):
        sys.stdout.flush()
        sys.stdout.write(f"\rTraining meta-epoch: {iteration}")
        
        meta_weights = REPTILE(ParamDict(meta_weights), gen_task)

        if iteration == 1 or iteration % PLOT_EVERY == 0:
            model = Model(meta_weights).cuda()
            model.train()  # set train mode
            opt = SGD(model.parameters(), lr=LR)

            for e in range(TEST_EPOCHS):
                loss = train_epoch(test_task, model, opt)
                sys.stdout.flush()
                sys.stdout.write(f"\rLoss in test task (epoch {e}): {loss}")

            evaluation(model, test_task, "Test Task")
            print("\n")
            # print(f"Iteration: {iteration}\tLoss: {evaluate(model, plot_task):.3f}")
