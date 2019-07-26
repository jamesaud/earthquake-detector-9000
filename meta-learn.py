#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from loaders.base_loader import SpectrogramBaseDataset
from torch import Tensor, linspace, nn, randperm, sin
from torch.autograd import Variable
from torch.nn import Linear
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from reptile.utils import ParamDict
from models import mnist_three_component_exp
from main import create_dataset, create_loader
import config 
from utils import dotdict, reduce_dataset, subset
import glob 
import os 
import copy
from typing import List, Callable
from pytorch_utils.utils import evaluation, save_checkpoint, load_checkpoint, write_evaluator
from pytorch_utils.utils import train as train_model

import sys
from datetime import datetime
from itertools import cycle
from writer_util import MySummaryWriter as SummaryWriter
import statistics

pj = os.path.join

configuration = 'meta_learning'

# Generate new settings when needed for creating different data loaders for each station... 
_settings = config.options[configuration]
_new_settings = lambda: dotdict(copy.deepcopy(_settings))
settings = _new_settings()


MODEL = mnist_three_component_exp

Weights = ParamDict
criterion = nn.CrossEntropyLoss().cuda() #F.l1_loss

PLOT = True
NUM_OF_TRAIN_STATIONS = None # all of them

LR, META_LR = 0.02, 0.1  # Copy OpenAI's hyperparameters.

# Training 
SAMPLES_PER_TASK = 100    # Samples for each station: 50
BATCH_SIZE, META_BATCH_SIZE = 10, 3
EPOCHS, META_EPOCHS = 1, 30000

# Evaluation
EVALUATION_TRAIN_SAMPLES = SAMPLES_PER_TASK
EVALUATION_EVAL_SAMPLES = SAMPLES_PER_TASK * 3
EVALUATION_EPOCHS = 5 
EVALUATE_EVERY = 10         # Also evaluates every 

# Test
TEST_TRAIN_SAMPLES = 100
TEST_EVALUATION_SAMPLES = 1000
TEST_EPOCHS = 20
TEST_EVERY = 30

path = os.path.join(os.path.join(config.VISUALIZE_PATH, f'runs/meta-learning/train/trial-{datetime.now()}'))
CHECKPOINT_PATH = os.path.join(path, 'checkpoints')
writer = SummaryWriter(path)


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

    def __init__(self, station_paths: List[str], task_samples:int, batch_size:int, train:bool=True, shuffle:bool=False, station_settings:dict=None):
        """
        :station_paths: 
        :task_samples:
        :batch_size:
        :train:
        :shuffle: randomly shuffles samples in each dataset (note, will be the same every time because random seed is set)
        """
        self.station_paths = station_paths
        self.train = train
        self.task_samples = task_samples
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.station_settings = station_settings or _new_settings()

        # Set up the datasets
        self._datasets = self._station_datasets(self.station_paths)
        self.iter_datasets = cycle(iter(self._datasets))

    def __call__(self) -> DataLoader:
        """
        Creates a new task loader by cycling through the datasets.
        """
        datasetIndex = next(self.iter_datasets)
        dataset, start_index = datasetIndex
        end_task_index = start_index + self.task_samples

        task_dataset = subset(dataset, range(start_index, end_task_index))
        task_loader = self.task_loader(task_dataset)

        datasetIndex[1] = len(dataset) % end_task_index
        return task_loader

    def skip(self, n):
        """ Skips a certain amount of __calls__ """
        for _ in range(n):
            self.__call__()
        return self

    def task_loader(self, dataset: Dataset) -> DataLoader:
        """
        Returns new task from the dataset
        """
        loader = create_loader(dataset, train=self.train, weigh_classes=self.station_settings.weigh_classes, batch_size=self.batch_size) 
        return loader

    def _create_dataset(self, station_path: str) -> Dataset:
        station_settings = _new_settings()

        if self.train:
            station_settings.train.path = station_path
        else:
            station_settings.test.path = station_path

        dataset = create_dataset(station_settings, Model.transformations['train' if self.train else 'test'], train=self.train)
        dataset.station_path = station_path 
        
        if self.shuffle:
            dataset.shuffle()

        return dataset

    def _station_datasets(self, station_paths: List[str]):
        # If not printing, can replace with just 1 line
        #datasets = ([self._create_dataset(path), 0] for path in station_paths)

        datasets = []
        n = len(station_paths)
        for i, path in enumerate(station_paths):
            sys.stdout.flush()
            sys.stdout.write(f"\r...creating dataset [{i + 1} of {n}]")

            # Use list, a mutable data structure, to store the Dataset and Current Index
            datasets.append([self._create_dataset(path), 0])
        print()
        return datasets


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


def train_epoch(task, model, opt):
    cum_loss = 0
    for x, y in task:
        loss = train_batch(x, y, model, opt)
        cum_loss += float(loss.data)
    return cum_loss


def train_epochs(task, model, opt, epochs):
    loss_per_epoch = [train_epoch(task, model, opt) for _ in range(epochs)]
    return loss_per_epoch


def sgd(meta_weights: Weights, epochs: int, task: DataLoader) -> Weights:
    """Run SGD on a randomly generated task."""
    model = Model(meta_weights).cuda()
    model.train()  # Ensure model is in train mode.
    opt = SGD(model.parameters(), lr=LR)
    loss = train_epochs(task, model, opt, epochs)[-1]   # Last epoch's loss
    return model.state_dict(), loss


def REPTILE(meta_weights: Weights,
            gen_task: Callable,
            meta_batch_size: int = META_BATCH_SIZE, 
            epochs: int = EPOCHS) -> Weights:
    """Run one iteration of REPTILE."""

    # Important for gen_task to be in the comprehension, so that a unique task is created each time 
    weights, losses = zip(*[sgd(meta_weights, epochs, gen_task()) for _ in range(meta_batch_size)])
    avg_loss = statistics.mean(losses)     # Avg loss of tasks                 

    weights = [ParamDict(w) for w in weights]

    # TODO Implement custom optimizer that makes this work with builtin
    # optimizers easily. The multiplication by 0 is to get a ParamDict of the
    # right size as the identity element for summation.
    meta_weights += (META_LR / epochs) * sum(
        (w - meta_weights for w in weights), 0 * meta_weights
    )
    return meta_weights, avg_loss



def make_checkpoint_name(evaluator, identifier):
    rounded = lambda decimal: str(round(decimal*100, 2))
    name = f"{identifier}" + \
            '-total-' + rounded(evaluator.total_percent_correct())+ \
            '-class0-' + rounded(evaluator.percent_correct(0)) + \
            '-class1-' + rounded(evaluator.percent_correct(1)) + \
            '.pt'
    return name


def evaluate_task(epochs, train_task, eval_task, model, opt, description, copy=True, writer=None, iteration=None):
    """ 
    Evaluate task, making sure not to re-train the current model.
    """
    if copy:
        model = copy.deepcopy(model)
        opt = copy.deepcopy(opt)

    for e in range(epochs):
        loss = train_epoch(train_task, model, opt)
        sys.stdout.flush()
        sys.stdout.write(f"\rLoss in train task (epoch {e}): {loss}")
    
    if writer:
        if iteration is None:
            raise ValueError("Must give iteration if providing writer")
        writer.add_scalar(f"{description} training loss", loss, iteration)

    model.eval()
    evaluator = evaluation(model, eval_task, description, copy_net=False)
    model.train()
    return evaluator


def write_info(writer):
    train_info = {
        'stations': NUM_OF_TRAIN_STATIONS,
        'epoch': EPOCHS,
        'meta_epochs': META_EPOCHS,
        'batch_size': BATCH_SIZE,
        'meta_batch_size': META_BATCH_SIZE,
        'train_samples_per_task': SAMPLES_PER_TASK,
        'learning_rate': LR,
        'meta_learning_rate': META_LR
    }

    eval_info = {
        'evaluation_train_samples': EVALUATION_TRAIN_SAMPLES,
        'evaluation_eval_samples': EVALUATION_EVAL_SAMPLES,
        'evaluation_epochs': EVALUATION_EPOCHS,
    }

    test_info = {
        'test_training_samples': TEST_TRAIN_SAMPLES,
        'test_eval_samples': TEST_EVALUATION_SAMPLES,
        'test_epochs': TEST_EPOCHS,
    }

    for step, (name, val) in enumerate(train_info.items()):
        writer.add_text('Train Settings', f"{name} = {val}", global_step=step)

    for step, (name, val) in enumerate(eval_info.items()):
        writer.add_text('Evaluation Settings', f"{name} = {val}", global_step=step)

    for step, (name, val) in enumerate(test_info.items()):
        writer.add_text('Test Settings', f"{name} = {val}", global_step=step)


if __name__ == "__main__":
    TRAIN = False
    TEST = not TRAIN

    station_paths = glob.glob(pj(settings.train.path, '*'))
    test_path = settings.test.path                   # Benz station
    station_paths.remove(test_path)

    if TRAIN:
        write_info(writer)

        if NUM_OF_TRAIN_STATIONS is None:
            NUM_OF_TRAIN_STATIONS = len(station_paths)

        print(f"Generating training tasks from {NUM_OF_TRAIN_STATIONS} stations:")
        gen_task = TaskCreator(station_paths[:NUM_OF_TRAIN_STATIONS], 
                            task_samples=SAMPLES_PER_TASK, 
                            batch_size=BATCH_SIZE, 
                            train=True)     # Training task callable

        print("Generating evaluation (train and eval) tasks...")
        evaluation_train_task = TaskCreator([test_path], 
                                            task_samples=EVALUATION_TRAIN_SAMPLES, 
                                            batch_size=BATCH_SIZE,
                                            train=True)()   # Evaluation task

        evaluation_eval_task = TaskCreator([test_path], 
                                            task_samples=EVALUATION_EVAL_SAMPLES, 
                                            batch_size=BATCH_SIZE,
                                            train=False)()

        print("Generating test (train and eval) tasks...")

        # Skip to be different than evaluation   
        test_train_task = TaskCreator([test_path], 
                                task_samples=TEST_TRAIN_SAMPLES, 
                                batch_size=BATCH_SIZE,
                                train=True).skip(3)()

        
        test_eval_task = TaskCreator([test_path], 
                                task_samples=TEST_EVALUATION_SAMPLES, 
                                batch_size=BATCH_SIZE,
                                train=False).skip(3)()


        # Need to put model on GPU first for tensors to have the right type.
        meta_weights = Model().cuda().state_dict()

        for iteration in range(1, META_EPOCHS + 1):
            sys.stdout.flush()
            sys.stdout.write(f"\rTraining meta-epoch: {iteration}")
            
            meta_weights, avg_loss = REPTILE(ParamDict(meta_weights), gen_task)
            writer.add_scalar('train_loss', avg_loss, iteration)  # Average train loss of the last epoch per task

            def new_meta_model():
                model = Model(meta_weights).cuda()
                model.train()
                opt = SGD(model.parameters(), lr=LR)
                return model, opt

            if iteration == 1 or iteration % EVALUATE_EVERY == 0:
                print()
                model, opt = new_meta_model()

                evaluator = evaluate_task(EVALUATION_EPOCHS, evaluation_train_task, evaluation_eval_task, model, opt, "Eval Task", copy=False, writer=writer, iteration=iteration)            

                name = make_checkpoint_name(evaluator, f"metaepoch-{iteration}")
                checkpoint_path = pj(CHECKPOINT_PATH, name)
                print(f"Saving checkpoint: {name}")
                save_checkpoint(checkpoint_path, name, model, opt, loss=0)   

                write_evaluator(writer, f'Evaluation Task', evaluator, iteration)            
                print('\n')

            if iteration == 1 or iteration % TEST_EVERY == 0:
                print()
                model, opt = new_meta_model()
                evaluator = evaluate_task(TEST_EPOCHS, test_train_task, test_eval_task, model, opt, "Test Task", copy=False, writer=writer, iteration=iteration)

                write_evaluator(writer, f'Test Task', evaluator, iteration)            
                print('\n')


    if TEST:
        model_name = 'metaepoch-670-total-92.0-class0-92.51-class1-91.15.pt'
        checkpoint_path = pj(config.VISUALIZE_PATH, f'runs/meta-learning/train/long_run/checkpoints/{model_name}')

        test_train_samples = 800
        test_eval_samples = 1000
        writer = SummaryWriter(path.replace('train', 'test'))
        
        test_station_settings = _new_settings()
        test_station_settings.weigh_classes = {0: 1, 1: 3}

        print("Creating Train and Test Loaders")
        test_train_loader = TaskCreator([test_path], 
                            task_samples=test_train_samples, 
                            batch_size=BATCH_SIZE,
                            train=True,
                            shuffle=True)()

    
        test_eval_loader = TaskCreator([test_path], 
                            task_samples=test_eval_samples, 
                            batch_size=BATCH_SIZE,
                            train=False,
                            shuffle=True)()

        model = Model().cuda()
        opt = SGD(model.parameters(), lr=LR)

        model, opt = load_checkpoint(checkpoint_path, model, opt, copy=False)
        model.train()

        def train_net(epochs):
            for i in range(epochs):
                train_model(i+1,
                            test_train_loader,
                            test_eval_loader,
                            opt,
                            criterion,
                            model,
                            writer,
                            write=True,
                            print_test_evaluation_every=90,
                            print_loss_every=10)

        train_net(100)

