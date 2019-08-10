import matplotlib
from pytorch_utils.utils import evaluate, write_images, load_model, print_evaluation, write_info, train
matplotlib.use('agg')
matplotlib.interactive(False)

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
from config import loader_args
 
if not torch.__version__.startswith("0.3"):
    print(f"PyTorch version should be 0.3.x, your version is {torch.__version__}.")
    exit()

configuration = 'benz_train_set'
settings = config.options[configuration]
settings = dotdict(settings)

CWD = os.getcwd()

# Train and test
make_path = lambda path: os.path.join(CWD, os.path.join('data', path))
 
# Variables
BATCH_SIZE = 128
NUM_CLASSES = 2
iterations = 0

WEIGH_CLASSES = settings.weigh_classes

# Neural Net Model
NET = models.mnist_three_component_exp

# Visualize
path = os.path.join(os.path.join(config.VISUALIZE_PATH, f'runs/{NET.__name__}/trial-{datetime.now()}'))
checkpoint_path = os.path.join(path, 'checkpoints')
writer = SummaryWriter(path)


# DATASET
loaders = {
    'direct': SpectrogramDirectDataset,
    'custom': SpectrogramCustomPathDataset,
    'named_timestamp': SpectrogramNamedTimestampDataset,
    'named': SpectrogramNamedDataset
}



# Added
def create_dataset(settings: dict, transformations, train: bool):
    TRAIN_IMG_PATH = make_path(settings.train.path)
    TEST_IMG_PATH = make_path(settings.test.path)

    Dataset = loaders[settings.loader]

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


    dataset_args = dict(
        path_pattern=settings.path_pattern or '',
        crop=crop,
        resize=resize
    )

    if train:
        dataset_train = Dataset(img_path=TRAIN_IMG_PATH,
                        transform=transformations,
                        ignore=settings.train.get('ignore'),
                        divide_test=settings.train.divide_test,
                        crop_padding=crop_padding_train,
                        **dataset_args
                        )
        return dataset_train

    else:
        dataset_test = Dataset(img_path=TEST_IMG_PATH,
                       transform=transformations,
                       ignore=settings.test.get('ignore'),
                       divide_test=settings.test.divide_test,
                       test=True,
                       crop_padding=crop_padding_test,
                       crop_center=False,
                       **dataset_args)
        return dataset_test

def create_loader(dataset, train: bool, batch_size = BATCH_SIZE, weigh_classes = None):
    num_classes = dataset.num_classes
    if train:
        train_sampler = utils.make_weighted_sampler(dataset, num_classes, weigh_classes=weigh_classes) if weigh_classes else None
        train_loader = DataLoader(dataset,
                          shuffle=not train_sampler,
                          sampler=train_sampler,
                          drop_last=False,
                          batch_size=batch_size,
                          **loader_args)
        
        return train_loader
    else:
        test_loader = DataLoader(dataset,
                          drop_last=False,
                          batch_size=batch_size,
                          **loader_args)
        return test_loader


# ADDED
def _main_make_datasets():
    dataset_train = create_dataset(settings, NET.transformations['train'], train=True)
    dataset_test = create_dataset(settings, NET.transformations['test'], train=False)
    assert verify_dataset_integrity(dataset_train, dataset_test)
    return dataset_train, dataset_test


def _main_make_loaders():
    dataset_train, dataset_test = _main_make_datasets()
    train_loader = create_loader(dataset_train, train=True, weigh_classes=WEIGH_CLASSES)
    test_loader = create_loader(dataset_test, train=False)
    return train_loader, test_loader



# Setup Net

def create_model():
    net = NET().cuda()
    optimizer = optim.Adam(net.parameters())
    criterion = nn.CrossEntropyLoss().cuda()
    return net, optimizer, criterion


def print_config():
    print(f"Using config {configuration}")
    pprint(settings)


def write_initial(writer, net, settings, datset_train):
    print("\nWriting Info")
    height_percent, width_percent = settings.image.crop or (1, 1)  # 0.5 to 1.0
    height, width = settings.image.height, settings.image.width
    resize = (height, width)
    crop = (int(height * height_percent), int(width * width_percent))

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

    net, optimizer, criterion = create_model()
    
    ################
    # TRAINING NEW MODEL
    #################

    # Subsample to evaluate train accuracy... because it has too many samples and will take too long
    num_train_evaluation_samples = 1000
    num_train_samples = 80
    num_test_samples = 1000


    dataset_train, dataset_test = _main_make_datasets()
    dataset_train = reduce_dataset(dataset_train, num_train_samples)
    dataset_test = reduce_dataset(dataset_test, num_test_samples)

    train_loader = create_loader(dataset_train, train=True, weigh_classes=WEIGH_CLASSES)
    test_loader = create_loader(dataset_test, train=False)

    train_evaluation_loader = DataLoader(reduce_dataset(dataset_train, num_train_evaluation_samples), **loader_args)


    if TRAIN_MODE:
        write_initial(writer, net, settings, dataset_train)

        def train_net(epochs):
            for epoch in range(epochs):
                train(epoch + 1, train_loader, test_loader, optimizer, criterion, net, writer,
                      write=True,
                      checkpoint_path=checkpoint_path,
                      print_test_evaluation_every=2_000,  # 30,000
                      print_train_evaluation_every=3_000,
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
            
