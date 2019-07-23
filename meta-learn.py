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
from pytorch_utils.utils import evaluation, save_checkpoint
import sys
from datetime import datetime


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
NUM_OF_TRAIN_STATIONS = 10
SAMPLES_PER_TASK = 50    # Samples for each station: 50
EVALUATION_TRAIN_SAMPLES = 50
EVALUATION_EVAL_SAMPLES = 50
TEST_SAMPLES = 1000

LR, META_LR = 0.02, 0.1  # Copy OpenAI's hyperparameters.
BATCH_SIZE, META_BATCH_SIZE = 10, 3
EPOCHS, META_EPOCHS = 1, 30000
EVALUATION_EPOCHS = 10
PLOT_EVERY = 10      
CHECKPOINT_PATH = f'./visualize/meta-models/trial-{datetime.now()}'


class WeightsModel(nn.Module):
    def __init__(self, weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if weights is not None:
            self.load_state_dict(deepcopy(weights))


class Model(WeightsModel, MODEL):
    """ 
    Allow for passing of weights as the __init__ argument for the chosen MODEL.
    """
    pass


class TaskCreator:
    """
    Class to create tasks given a list of station paths.
    Tasks are dataloaders, with references kept in self.station_loaders
    """
    def __init__(self, station_paths: List, num_samples=None, train=True):
        self.station_paths = station_paths
        self.station_loaders = self._station_loaders(station_paths, num_samples, train)
        self.__call__index = 0

    def __call__(self) -> DataLoader:
        """
        Cycles through station loaders everytime it is called, like an infinite generator (but not implemented as __iter__)
        """
        i = self.__call__index
        self._next__call__index()
        loader = self.station_loaders[i]
        return self.station_loaders[i]

    def _next__call__index(self):
        self.__call__index = len(self.station_loaders) % (self.__call__index + 1)

    @staticmethod
    def _station_loaders(station_paths, num_samples, train):
        return [station_loader(path, num_samples=num_samples, train=train) for path in station_paths]
    

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

def sgd(meta_weights: Weights, epochs: int, task: DataLoader) -> Weights:
    """Run SGD on a randomly generated task."""
    model = Model(meta_weights).cuda()
    model.train()  # Ensure model is in train mode.
    opt = SGD(model.parameters(), lr=LR)
    train_epochs(task, model, opt, epochs)
    return model.state_dict()


def REPTILE(meta_weights: Weights,
            gen_task: Callable,
            meta_batch_size: int = META_BATCH_SIZE, 
            epochs: int = EPOCHS) -> Weights:
    """Run one iteration of REPTILE."""

    # Important for gen_task to be in the comprehension, so that a unique task is created each time 
    weights = [sgd(meta_weights, epochs, gen_task()) for _ in range(meta_batch_size)]

    weights = [ParamDict(w) for w in weights]

    # TODO Implement custom optimizer that makes this work with builtin
    # optimizers easily. The multiplication by 0 is to get a ParamDict of the
    # right size as the identity element for summation.
    meta_weights += (META_LR / epochs) * sum(
        (w - meta_weights for w in weights), 0 * meta_weights
    )
    return meta_weights


def station_loader(station_path: str, num_samples=None, train=True) -> DataLoader:
    station_settings = _new_settings()

    if train:
        station_settings.train.path = station_path
    else:
        station_settings.test.path = station_path

    dataset = create_dataset(station_settings, Model.transformations['train' if train else 'test'], train=train)
    if num_samples:
        reduce_dataset(dataset, num_samples, copy_dataset=False)

    loader = create_loader(dataset, train=train, weigh_classes=station_settings.weigh_classes, batch_size=BATCH_SIZE) 
    return loader

def make_checkpoint_name(evaluator, identifier):
    rounded = lambda decimal: str(round(decimal*100, 2))
    name = f"{identifier}" + \
            '-total-' + rounded(evaluator.total_percent_correct())+ \
            '-class0-' + rounded(evaluator.percent_correct(0)) + \
            '-class1-' + rounded(evaluator.percent_correct(1)) + \
            '.pt'
    return name


def evaluate_task(epochs, train_task, eval_task, model, opt):
    """ 
    Evaluate task, making sure not to re-train the current model.
    """
    model = copy.deepcopy(model)
    opt = copy.deepcopy(opt)

    for e in range(epochs):
        loss = train_epoch(train_task, model, opt)
        sys.stdout.flush()
        sys.stdout.write(f"\rLoss in train task (epoch {e}): {loss}")

    evaluator = evaluation(model, eval_task, "Loss in Eval Task")
    return evaluator

if __name__ == "__main__":
    #TODO: Change the training loaders to load new samples every time they are called 

    station_paths = glob.glob(pj(settings.train.path, '*'))
    test_path = settings.test.path                   # Benz station
    station_paths.remove(test_path)
    
    print(f"Generating {NUM_OF_TRAIN_STATIONS} training tasks...")
    gen_task = TaskCreator(station_paths[:NUM_OF_TRAIN_STATIONS], num_samples=SAMPLES_PER_TASK, train=True)     # Training task callable

    print("Generating evaluation (train and eval) tasks...")
    evaluation_train_task = TaskCreator([test_path], num_samples=EVALUATION_TRAIN_SAMPLES, train=True)()   # Evaluation task
    evaluation_eval_task = TaskCreator([test_path], num_samples=EVALUATION_EVAL_SAMPLES, train=False)()

    print("Generating test task...")
    test_task = TaskCreator([test_path], num_samples=TEST_SAMPLES, train=False)()

    # Need to put model on GPU first for tensors to have the right type.
    meta_weights = Model().cuda().state_dict()

    for iteration in range(1, META_EPOCHS + 1):
        sys.stdout.flush()
        sys.stdout.write(f"\rTraining meta-epoch: {iteration}")
        
        meta_weights = REPTILE(ParamDict(meta_weights), gen_task)

        if iteration == 1 or iteration % PLOT_EVERY == 0:
            print()
            model = Model(meta_weights).cuda()
            model.train()  # set train mode
            opt = SGD(model.parameters(), lr=LR)

            print("\nEVALUATION TASK PERFORMANCE ")
            evaluator = evaluate_task(EVALUATION_EPOCHS, evaluation_train_task, evaluation_eval_task, model, opt)

            print("\nTEST TASK PERFORMANCE")
            evaluate_task(EVALUATION_EPOCHS, evaluation_train_task, test_task, model, opt)

            name = make_checkpoint_name(evaluator, f"metaepoch-{iteration}")
            checkpoint_path = pj(CHECKPOINT_PATH, name)
            print(f"Saving checkpoint: {name}")
            save_checkpoint(checkpoint_path, name, model, opt, loss=0)    
            print('\n')

                
