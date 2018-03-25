
from loaders.single_loader import SpectrogramDataset
from loaders.multiple_loader import Spectrogram3ComponentDataset
import models
from matplotlib import pyplot as plt
import numpy as np


plt.switch_backend("TkAgg")

IMG_PATH = './spectrograms/'
IMG_EXT = '.png'
BATCH_SIZE = 256

# variables
NET = models.mnist_multiple
MODEL_PATH = f'checkpoints/{NET.__name__}'


# Dataset



def preview_multiple_loader_dataset(num):
    dataset_train = Spectrogram3ComponentDataset(IMG_PATH, transform=NET.transformations['train'])
    dt = iter(dataset_train)
    imgs, label = next(dt)

    dataset_train.preview(index=num, show=False)

# Good Local Indexes: 3, 13
preview_multiple_loader_dataset(13)
plt.show()



def compute_mean_and_std():
    dataset_train = SpectrogramDataset(IMG_PATH, transform=NET.transformations['train'])

    r, g, b = [], [], []
    for img, label in dataset_train:
        img = img.numpy()
        r.append(img[0]); g.append(img[1]); b.append(img[2])

    r, g, b = np.array(r).flatten(), np.array(g).flatten(), np.array(b).flatten()

    means = [color.mean()/255 for color in (r, g, b)]
    stds = [color.std()/255 for color in (r, g, b)]
    print(means, stds)

#compute_mean_and_std()


# GRAY Mean: [0.0007967819185817943, 0.0007967819185817943, 0.0007967819185817943]
# GRAY std : [0.0002987987562721851, 0.0002987987562721851, 0.0002987987562721851]

# GRAY Mean: [0.0007967819185817943, 0.0007967819185817943, 0.0007967819185817943]
# GRAY std : [0.0002987987562721851, 0.0002987987562721851, 0.0002987987562721851]



# Sampler
# indices = list(range(len(dataset_train)))
# np.random.shuffle(indices)
# test_split = 3000
#
# train_idx, test_idx = indices[test_split:], indices[:test_split]
#
# train_sampler = SubsetRandomSampler(train_idx)
# test_sampler = SubsetRandomSampler(test_idx)


