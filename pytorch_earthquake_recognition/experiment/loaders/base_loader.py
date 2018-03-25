from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import glob
from matplotlib import pyplot as plt
import random
import os
from collections import namedtuple
from PIL import ImageFile
from mytransforms.RandomSameCrop import RandomSameCropWidth
from typing import List
from PIL import Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
Components = namedtuple('Components', ('N', 'Z', 'E'))

class SpectrogramBaseDataset(Dataset):
    """
    """
    __SEED = 448

    def __init__(self, img_path, transform=None, test=False, resize=(200, 310), crop=(200, 260), divide_test=.3, **kwargs):
        """

        :param img_path: path to the 'spectrograms' folder
        :param transform: list of transforms to apply AFTER resize and crop
        :param test: boolean, training or test
        :param resize:  height, width or False
        :param crop:   height, width or False
        :param divide_test: 0 < divide_test < 1, percentage to divide test set into

        Order:
             - resize
             - crop
             - transform
        """

        self.crop, self.resize = crop, resize

        self.img_path = img_path
        self.test = test

        # Transforms
        self.transform = transform
        self.resize = transforms.Resize(resize)
        self.transform_random_same_crop = RandomSameCropWidth(crop[1])

        # Get paths
        local_paths, noise_paths = self.get_spectrograms(img_path)

        # Randomly shuffle the files the same way each time, to keep the test and train dataset the same
        self.shuffle(local_paths)
        self.shuffle(noise_paths)

        # Divide into training and test set
        test_local, train_local = self.separate_paths(local_paths, divide_test)
        test_noise, train_noise = self.separate_paths(noise_paths, divide_test)

        if test:
            file_paths = test_local + test_noise
        else:
            file_paths = train_local + train_noise

        self.file_paths = self.shuffle(file_paths)

        self.labels = {
            'noise': 0,
            'local': 1,
        }
        self.reverse_map(self.labels)

    def separate_paths(self, paths, amount):
        if not (0 < amount < 1):
            raise ValueError("'amount' should be between 0 and 1")
        separate_index = int(len(paths) * amount)
        return paths[:separate_index], paths[separate_index:]

    def shuffle(self, paths):
        random.seed(self.__SEED)
        random.shuffle(paths)
        return paths

    @staticmethod
    def get_spectrograms(path):
        local_path = glob.glob(os.path.join(path + '/*/local/*/'))
        noise_path = glob.glob(os.path.join(path + '/*/noise/*/'))

        def get_components(paths_to_components):
            return [glob.glob(os.path.join(path, "*.png")) for path in paths_to_components]

        local_paths = get_components(local_path)
        noise_paths = get_components(noise_path)

        return local_paths, noise_paths

    def apply_transforms(self, components):
        components = [self.open_image(component) for component in components]

        if self.resize:
            components = self.apply_resize(components)

        if self.crop:
            components = self.apply_crop(components)

        if self.transform:
            components = map(self.transform, components)

        n, z, e = components
        return n, z, e

    @staticmethod
    def open_image(img_path):
        img = Image.open(img_path)
        img = img.convert('RGB')
        return img

    def apply_crop(self, components):
        components[0].height
        if self.test:
            crop = transforms.CenterCrop(self.crop)
        else:
            self.transform_random_same_crop.set_params(components[0], self.crop[1])
            crop = self.transform_random_same_crop
        n, z, e = map(crop, components)
        return n, z, e

    def apply_resize(self, components):
        n, z, e = map(self.resize, components)
        return n, z, e

    def __getitem__(self, index):
        n, z, e = self.file_paths[index]
        label = self.label_to_number(self.get_label(n))
        n, z, e = self.apply_transforms((n, z, e))
        return Components(n, z, e), label

    def __len__(self):
        return len(self.file_paths)

    def label_to_number(self, label):
        return self.labels[label]

    def preview(self, index=0, show=True):
        components, label = self.__getitem__(index)

        fig = plt.figure()
        plt.suptitle(self.labels[label])
        for i, title in enumerate(['n', 'z', 'e']):
            ax = plt.subplot(1, 3, i + 1)
            plt.tight_layout()
            ax.set_title(title)
            ax.axis('off')
            self.show_img(transforms.ToPILImage()(components[i]))

        if show:
            plt.show()

    @staticmethod
    def clean_paths(file_paths):
        return [components for components in file_paths if len(components) == 3]

    @staticmethod
    def reverse_map(dic):
        dic.update({v:k for k,v in dic.items()})

    @staticmethod
    def get_label(file_path):
        return file_path.split('/')[-3]

    @staticmethod
    def show_img(image):
        plt.imshow(image)
        plt.pause(0.001)


if __name__ == '__main__':
    IMG_PATH = '../spectrograms'
    s = SpectrogramBaseDataset(IMG_PATH)
    s = iter(s)
    print(next(s))