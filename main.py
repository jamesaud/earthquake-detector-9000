import matplotlib
from pytorch_utils.utils import evaluate, write_images, load_model, print_evaluation, write_info, train
matplotlib.use('agg')
matplotlib.interactive(False)

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loaders.single_loader import SpectrogramSingleDataset
from loaders.custom_path_loader import SpectrogramCustomPathDataset
from loaders.direct_loader import SpectrogramDirectDataset
from loaders.named_loader import SpectrogramNamedDataset, SpectrogramNamedTimestampDataset
import models
import os
from datetime import datetime
import config
from writer_util import MySummaryWriter as SummaryWriter
from utils import dotdict, verify_dataset_integrity, reduce_dataset, subsample_dataset
import utils
from pprint import pprint
from evaluator.csv_write import write_named_predictions_to_csv
from writer_util.stats_writer import StatsWriter
import copy
 
if not torch.__version__.startswith("0.3"):
    print(f"PyTorch version should be 0.3.x, your version is {torch.__version__}.")
    exit()

configuration = 'benz_train_set'
settings = config.options[configuration]
settings = dotdict(settings)

CWD = os.getcwd()

# Train and test
make_path = lambda path: os.path.join(CWD, os.path.join('data', path))

TRAIN_IMG_PATH = make_path(settings.train.path)
TEST_IMG_PATH = make_path(settings.test.path)
 
# Variables
BATCH_SIZE = 128
NUM_CLASSES = 2
iterations = 0

WEIGH_CLASSES = settings.weigh_classes

# Neural Net Model
NET = models.mnist_one_component

# Visualize
path = os.path.join(os.path.join(config.VISUALIZE_PATH, f'runs/{NET.__name__}/trial-{datetime.now()}'))
checkpoint_path = os.path.join(path, 'checkpoints')
writer = SummaryWriter(path)

# Dimentional Transforms
height_percent, width_percent = settings.image.crop or (1, 1)  # 0.5 to 1.0
height, width = settings.image.height, settings.image.width

resize = (height, width)
crop = (int(height * height_percent), int(width * width_percent))

crop_padding_train = settings.image.padding_train
crop_padding_test = settings.image.padding_test

if crop_padding_train:
    crop_padding_train = utils.calculate_crop_padding_pixels(crop_padding_train, height, width)

if crop_padding_test:
    crop_padding_test = utils.calculate_crop_padding_pixels(crop_padding_test, height, width)

# DATASET
loaders = {
    'direct': SpectrogramDirectDataset,
    'custom': SpectrogramCustomPathDataset,
    'named_timestamp': SpectrogramNamedTimestampDataset,
    'named': SpectrogramNamedDataset
}

Dataset = SpectrogramSingleDataset#loaders[settings.loader]

dataset_args = dict(
    path_pattern=settings.path_pattern or '',
    crop=crop,
    resize=resize
    )


dataset_train = Dataset(img_path=TRAIN_IMG_PATH,
                        transform=NET.transformations['train'],
                        ignore=settings.train.get('ignore'),
                        divide_test=settings.train.divide_test,
                        crop_padding=crop_padding_train,
                        **dataset_args
                        )

print(next(iter(dataset_train)))

dataset_test = Dataset(img_path=TEST_IMG_PATH,
                       transform=NET.transformations['test'],
                       ignore=settings.test.get('ignore'),
                       divide_test=settings.test.divide_test,
                       test=True,
                       crop_padding=crop_padding_test,
                       crop_center=False,
                       **dataset_args)

assert verify_dataset_integrity(dataset_train, dataset_test)

train_sampler = utils.make_weighted_sampler(dataset_train, NUM_CLASSES, weigh_classes=WEIGH_CLASSES) if WEIGH_CLASSES else None

# Data Loaders
loader_args = dict(
                   batch_size=BATCH_SIZE,
                   num_workers=8,
                   pin_memory=True,
                   )

train_loader = DataLoader(dataset_train,
                          shuffle=not train_sampler,
                          sampler=train_sampler,
                          drop_last=False,
                          **loader_args)


# Subsample to evaluate train accuracy... because it has too many samples and will take too long
num_train_evaluation_samples = 20000
train_evaluation_loader = DataLoader(reduce_dataset(dataset_train, num_train_evaluation_samples), **loader_args)

test_loader = DataLoader(dataset_test,
                         drop_last=False,
                         **loader_args)


# Setup Net
net = NET().cuda()
optimizer = optim.Adam(net.parameters())
criterion = nn.CrossEntropyLoss().cuda()


def print_config():
    print(f"Using config {configuration}")
    pprint(settings)


def write_initial(writer, net, settings, resize, crop, datset_train):
    print("\nWriting Info")
    write_info(writer, net, settings, resize, crop)
    write_images(writer, dataset_train)


def write_stats(evaluator, name):
    class_labels = {0: 'Noise', 1: "Event"}
    stats_writer = StatsWriter(os.path.join(CWD, f'visualize/test_stats/{name}'))
    softmax_output_labels = nn.functional.softmax(torch.autograd.Variable(evaluator.output_labels), dim=1).data   # Make probabilities sum between 0 and 1
    stats_writer.write_stats(evaluator.true_labels.numpy(),
                             softmax_output_labels.numpy(),
                             evaluator.predicted_labels,
                             class_labels)

if __name__ == '__main__':
    print_config()
    TRAIN_MODE = True
    
    ################
    # TRAINING NEW MODEL
    #################

    if TRAIN_MODE:
        write_initial(writer, net, settings, resize, crop, dataset_train)

        def train_net(epochs):
            for epoch in range(epochs):
                train(epoch + 1, train_loader, test_loader, optimizer, criterion, net, writer,
                      write=True,
                      checkpoint_path=checkpoint_path,
                      print_test_evaluation_every=60_000,  # 30,000
                      print_train_evaluation_every=100_000,
                      train_evaluation_loader=train_evaluation_loader
                      )

        train_net(60)

    ########################
    # TEST EXISTING MODEL
    #######################

    # Test Mode
    else:
        # Load Net
        MODEL = f'benz-training/checkpoints'
        model_name = "iterations-780416-total-99.07-class0-98.91-class1-99.37.pt"
        MODEL_PATH = f'./visualize/runs/{NET.__name__}/{MODEL}/{model_name}'

        # Set model to evaluation mode
        print("Loading Net")
        load_model(net, MODEL_PATH)
        net.eval()

        # Make compatible with functions that  expect loaders to return 2 items (if using namedloader)
        dataset_test.return_name = False

        # Test the evaluator (to see results output)
        print("Testing Net")
        test_evaluator = evaluate(net, test_loader, copy_net=True)
        print()
        print_evaluation(test_evaluator, 'test')

        # Write figures
        print("Writing stats...")
        write_stats(test_evaluator, f'{model_name}-{configuration}')

        # Compatible with csv write functions that expect 3 items (components, label, name)
        dataset_test.return_name = True

        # Write CSV predictions
        write_named_predictions_to_csv(net, test_loader, f'evaluator/predictions({model_name}-{configuration}).csv')
        print("\nWrote csv")
            
