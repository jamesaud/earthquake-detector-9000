import matplotlib
matplotlib.use('Agg')

from pathlib import Path
from torch.utils.data import DataLoader
from loaders import SpectrogramDirectDataset, SpectrogramCustomPathDataset, SpectrogramSingleDataset
import models
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms
from collections import Counter
import PIL

IMG_PATH = './data/Benz/spectrograms/test_set_benz_2'  # './data/Benz/spectrograms/train_set_benz'
IMG_EXT = '.png'
BATCH_SIZE = 256

# variables
NET = models.mnist_one_component
NET_MULTIPLE = models.mnist_three_component_exp
MODEL_PATH = f'checkpoints/{NET.__name__}'
write_path = Path('visualize/scratch/')

TRANSFORMS = transforms.Compose(
    [transforms.ToTensor()]
)

def preview_multiple_dataset(index=None, label=None, raw=False, shuffle=False):
    """
    :param num: index to look up
    :param label: should be 0 or 1. Don't provide num and label
    """
    if index is None and label is None:
        raise ValueError('index or label must be provided as an argument')

    if index and label:
        raise ValueError("can't use both index and label as arguments")

    dataset_train = SpectrogramDirectDataset(img_path=IMG_PATH,
                                             divide_test=0,
                                             transform=NET_MULTIPLE.transformations['train'],
                                             shuffle=shuffle)

    print(next(dataset_train.get_next_index_with_label('noise')))
    if index is None:
        index = next(dataset_train.get_next_index_with_label(label))

    print(index)

    if raw:
        return dataset_train.preview_raw(index=index, show=False)
    else:
        return dataset_train.preview(index=index, show=False)


def compute_mean_and_std(grayscale=False, samples=5000):

    dataset_train = SpectrogramSingleDataset(IMG_PATH,
                                             divide_test=0,
                                             shuffle=True)

    # Computationally intense, so only use a subset
    del dataset_train.file_paths[samples:]

    to_pil = transforms.ToPILImage()
    to_grayscale = transforms.Grayscale(num_output_channels=3)
    to_tensor = transforms.ToTensor()

    r, g, b = [], [], []

    for img, label in dataset_train:
        if grayscale:
            img = to_grayscale(img)
        img = to_tensor(img).numpy()
        r.append(img[0]);
        g.append(img[1]);
        b.append(img[2])

    r, g, b = np.array(r).flatten(), np.array(g).flatten(), np.array(b).flatten()

    means = [color.mean()/255 for color in (r, g, b)]
    stds = [color.std()/255 for color in (r, g, b)]
    print(means, stds)


def compute_mean_and_std_np(grayscale=False, samples=5000):
    dataset_train = SpectrogramSingleDataset(IMG_PATH,
                                             divide_test=0,
                                             shuffle=True)

    # Computationally intense, so only use a subset
    del dataset_train.file_paths[samples:]

    to_pil = transforms.ToPILImage()
    to_grayscale = transforms.Grayscale(num_output_channels=3)
    to_tensor = transforms.ToTensor()

    rgb = np.array([], dtype=int)

    for img, label in dataset_train:
        if grayscale:
            img = to_grayscale(img)
        img = to_tensor(img).numpy()
        np.concat(rgb, img)

    print(rgb)

    means = [color.mean() / 255 for color in (r, g, b)]
    stds = [color.std() / 255 for color in (r, g, b)]
    print(means, stds)


def img_border_color(img: PIL.Image):
    """
    Look at the 4 corner pixels of the image to determine the border color.
    They should all be the same to guarantee it's the correct color.
    """
    width, height = img.size
    bottom_left = (height, 0)
    return img.getpixel(bottom_left)


def find_border_colors(gray=False):
    """
    Finds the RGB and Gray border colors
    :return: int if grayscale, tuple (r: int, g: int, b: int) if rgb
    """
    transform = transforms.Grayscale(num_output_channels=3) if gray else None

    dataset_train = SpectrogramDirectDataset(img_path=IMG_PATH,
                                             divide_test=0,
                                             transform=transform,
                                             shuffle=True)

    iter_dataset = iter(dataset_train)
    imgs, labels = next(iter_dataset)

    for _ in iter_dataset:
        colors = list(map(img_border_color, imgs))
        print(colors)

    return colors[0]

if __name__ == '__main__':

    ## Mean and STD ##
    # compute_mean_and_std(grayscale=True)

    # RGB Mean: [0.0009950225259743484, 0.000795141388388241, 0.0018111652018977147]
    # RGB std : [0.0001881388618665583, 0.0006368028766968671, 0.00028853512862149407]

    # GRAY Mean: [0.0009636541207631429, 0.0009636541207631429, 0.0009636541207631429]
    # GRAY std : [0.0003674938398249009, 0.0003674938398249009, 0.0003674938398249009]


    ## BORDER COLOR CALCULATIONS

    rgb_color = find_border_colors()
    # gray_color = find_border_colors(gray=True)

    print("RGB border color", rgb_color)
    # print("Gray border color", gray_color)

    #
    # ## VIEW IMAGES ##

    # fig = preview_multiple_dataset(label=0, raw=True)
    # plt.savefig(write_path / 'fig_noise.png')
    #
    # fig = preview_multiple_dataset(label=1, raw=True)
    # plt.savefig(write_path / 'fig_event.png')