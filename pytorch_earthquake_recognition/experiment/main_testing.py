
from loaders.single_loader import SpectrogramSingleDataset
from loaders.multiple_loader import SpectrogramMultipleDataset
import models
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms

plt.switch_backend("TkAgg")

IMG_PATH = './spectrograms/'
IMG_EXT = '.png'
BATCH_SIZE = 256

# variables
NET = models.mnist_one_component
NET_MULTIPLE = models.mnist_three_component
MODEL_PATH = f'checkpoints/{NET.__name__}'


# Dataset
def preview_dataset(num, raw=False):
    dataset_train = SpectrogramSingleDataset(IMG_PATH, transform=NET.transformations['train'])
    if raw:
        dataset_train.preview_raw(index=num, show=False)
    else:
        dataset_train.preview(index=num, show=False)


def preview_multiple_dataset(num, raw=False):
    dataset_train = SpectrogramMultipleDataset(IMG_PATH, transform=NET_MULTIPLE.transformations['train'])
    if raw:
        dataset_train.preview_raw(index=num, show=False)
    else:
        dataset_train.preview(index=num, show=False)


# Good Local Indexes: 3, 13
preview_multiple_dataset(13, raw=True)
preview_multiple_dataset(13, raw=False)



def compute_mean_and_std(grayscale=False):
    dataset_train = SpectrogramSingleDataset(IMG_PATH)
    resize = transforms.Resize((217, 316))
    to_pil = transforms.ToPILImage()
    to_grayscale = transforms.Grayscale(num_output_channels=3)
    to_tensor = transforms.ToTensor()

    r, g, b = [], [], []

    for img, label in dataset_train:
        img = resize(img)
        if grayscale:
            img = to_grayscale(img)
        img = to_tensor(img)
        img = img.numpy()
        r.append(img[0]); g.append(img[1]); b.append(img[2])

    r, g, b = np.array(r).flatten(), np.array(g).flatten(), np.array(b).flatten()

    means = [color.mean()/255 for color in (r, g, b)]
    stds = [color.std()/255 for color in (r, g, b)]
    print(means, stds)



# compute_mean_and_std(grayscale=True)


# RGB Mean: [0.0009950225259743484, 0.000795141388388241, 0.0018111652018977147]
# RGB std : [0.0001881388618665583, 0.0006368028766968671, 0.00028853512862149407]

# GRAY Mean: [0.0009636541207631429, 0.0009636541207631429, 0.0009636541207631429]
# GRAY std : [0.0003674938398249009, 0.0003674938398249009, 0.0003674938398249009]



# Sampler
# indices = list(range(len(dataset_train)))
# np.random.shuffle(indices)
# test_split = 3000
#
# train_idx, test_idx = indices[test_split:], indices[:test_split]
#
# train_sampler = SubsetRandomSampler(train_idx)
# test_sampler = SubsetRandomSampler(test_idx)


