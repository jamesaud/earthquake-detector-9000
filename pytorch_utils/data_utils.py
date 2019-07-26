import copy
import torch
import math
from torch.utils.data.sampler import SequentialSampler, BatchSampler

def subsample_dataset(dataset: torch.utils.data.Dataset, num_samples, class_weights: dict, copy_dataset=True):
    if copy_dataset: dataset = copy.deepcopy(dataset)
    weight_sum = sum(class_weights.values())
    num_local = math.ceil(class_weights[1] / weight_sum * num_samples)
    num_noise = math.ceil(class_weights[0] / weight_sum * num_samples)
    dataset.file_paths = dataset.local[:num_local] + dataset.noise[:num_noise]
    return dataset


def replace_loader_dataset(dataloader: torch.utils.data.DataLoader, dataset: torch.utils.data.Dataset, sampler=None):
    dataloader.dataset = dataset
    if sampler is None:
        print(f"* Warning - sampler {dataloader.sampler.__class__.__name__} is being replaced by SequentialSampler *")
        sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, dataloader.batch_size, dataloader.drop_last)
    dataloader.batch_sampler = batch_sampler
