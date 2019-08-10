import torch
import sys
from torch.autograd import Variable
import torch.utils.data
from functools import wraps
import copy
import math
from torch.utils.data import Dataset
#from loaders import SpectrogramBaseDataset
from typing import Sequence
import random
random.seed(244)

class dotdict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    """dot.notation access to dictionary attributes"""
    def __getattr__(self, attr):
        return self.get(attr)

    __delattr__ = dict.__delitem__

    def __setitem__(self, key, value):
        if type(value) is dict:
            value = dotdict(value)
        return super().__setitem__(key, value)

    def get(self, key):
        value = super().get(key)
        if type(value) is dict:
            self.__setitem__(key, dotdict(value))
            value = self.__getitem__(key)
        return value


def make_weights_for_classes(images, nclasses, weigh_classes=None):
    count = [0] * nclasses   # how many of each class there are
    items = [images.__getitem__(index, apply_transforms=False) for index in range(len(images))]
    
    weigh_classes = dict(enumerate(weigh_classes)) if weigh_classes else dict()
    for item in items:
        label = item[1]
        count[label] += 1

    weight_per_class = [0.] * nclasses
    N = sum(count)   # Total number of samples

    for i in range(nclasses):
        if count[i] == 0:
            weight_per_class[i] = 0
        else:
            weight_per_class[i] = N/count[i]         

    weight = [0] * len(images)
    for idx, val in enumerate(items):
        label = val[1]
        weight[idx] = weight_per_class[label] * weigh_classes.get(label, 1)
    return weight


# Weighted sampler
def make_weighted_sampler(dataset, num_classes, weigh_classes=None) -> torch.utils.data.sampler.WeightedRandomSampler:
    weights = make_weights_for_classes(dataset, num_classes, weigh_classes)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    return sampler


class IntegrityError(Exception):
    pass

def verify_dataset_integrity(*args):
    paths = set()

    for dataset in args:
        for trio in dataset.file_paths:
            for path in trio:
                if path in paths:
                    raise IntegrityError(f'Duplicate file found in datasets "{path}"')
                paths.add(path)

    return True

def reduce_dataset(dataset: torch.utils.data.Dataset, num_samples, copy_dataset=True):
    if copy_dataset: dataset = copy.deepcopy(dataset)
    del dataset.file_paths[num_samples:]   
    return dataset

# dataset: SpectrogramBaseDataset
def subset(dataset, indices: Sequence, copy_dataset=True):
    if copy_dataset: dataset = copy.deepcopy(dataset)
    dataset.file_paths = [dataset.file_paths[i] for i in indices]
    return dataset

def subsample_dataset(dataset: torch.utils.data.Dataset, num_samples, class_weights: dict, random_shuffle=False, copy_dataset=True):
    if copy_dataset: dataset = copy.deepcopy(dataset)
    weight_sum = sum(class_weights.values())
    num_local = math.ceil(class_weights[1] / weight_sum * num_samples)
    num_noise = math.ceil(class_weights[0] / weight_sum * num_samples)

    if random_shuffle:
        dataset.file_paths = random.sample(dataset.local, num_local) + random.sample(dataset.noise, num_noise)
    else:
        dataset.file_paths = dataset.local[:num_local] + dataset.noise[:num_noise]

    dataset.shuffle()
    
    return dataset

def lmap(*args, **kwargs):
    return list(map(*args, **kwargs))


def calculate_crop_padding_pixels(crop_padding_percent, img_height, img_width):
    height, width = img_height, img_width
    left, right, top, bottom = crop_padding_percent
    left, right = width * left, width * right
    top, bottom = height * top, height * bottom
    return (left, right, top, bottom)      # (padding_left, padding_right, top, bottom) in pixels


from functools import wraps
from time import time
import sys

def timing(f, msg=None):
    i = 0

    @wraps(f)
    def wrap(*args, **kw):
        nonlocal i
        nonlocal msg
        i += 1
        ts = time()
        result = f(*args, **kw)
        te = time()
        if msg is not None:
            message = msg + f' | Progress({i})'
        else:
            message = 'func:%r args:[%r, %r] took: %2.4f sec | %r ' % (f.__name__, args, kw, te - ts, i)

        sys.stdout.write(f"\r {message}")
        sys.stdout.flush()
        return result
    return wrap

def timing_msg(msg):
    def timing_msg(fn):
        return timing(fn, msg=msg)
    return timing_msg


