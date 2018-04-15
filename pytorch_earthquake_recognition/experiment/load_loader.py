import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from loaders.multiple_loader import SpectrogramMultipleDataset
import models
import os
from datetime import datetime
import config
from utils import Evaluator
from writer_util import MySummaryWriter as SummaryWriter
from utils import dotdict
import sys
import utils

# Settings
ignore = config.Folders.values()
ignore.remove(config.Folders.Oklahoma.value)

options = dict(
    oklahoma={
        'train': {
            'path': 'spectrograms-oklahoma/Train',
            'divide_test': 0.2,
        },
        'test': {
            'path': 'spectrograms-oklahoma/Train',
            'divide_test': 0.2
        },
        'image': {
          'height': 258,
          'width': 293,
        }
    },
    custom={
        'train': {
            'path': 'spectrograms',
            'divide_test': 0.2,
            'ignore': ignore
        },
        'test': {
            'path': 'spectrograms',
            'divide_test': 0.2,
            'ignore': ignore
        },
        'image': {
          'height': 217,
          'width': 296,
        }
    }
)

settings = options['oklahoma']
settings = dotdict(settings)


# Train and test
make_path = lambda path: os.path.join(os.getcwd(), os.path.join('data', path))

TRAIN_IMG_PATH = make_path(settings.train.path)
TEST_IMG_PATH = make_path(settings.test.path)

# Variables
BATCH_SIZE = 256   # 128
NUM_CLASSES = 2
iterations = 0

# Neural Net Model
NET = models.mnist_three_component
MODEL_PATH = f'checkpoints/{NET.__name__}'

# Visualize
path = os.path.join(os.path.join(config.VISUALIZE_PATH, f'runs/{NET.__name__}/trial-{datetime.now()}'))
writer = SummaryWriter(path)

# Ignore Paths
ignore_train = settings.train.get('ignore')
ignore_test = settings.test.get('ignore')

# Dataset
train_test_split = settings.train.divide_test
test_split = settings.test.divide_test

# Dimentional Transforms
width_percent = 1  # 0.5 to 1.0
height, width = settings.image.height, settings.image.width

resize = (height, width)   # (height, width)
crop = (height, int(width * width_percent))     # (height, width)


