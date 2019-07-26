from torch.utils.data.dataset import Dataset
from torchvision import transforms
import glob
from matplotlib import pyplot as plt
import random
import os
from collections import namedtuple
from PIL import ImageFile
from mytransforms import RandomSameCropWidth, RandomSameCrop
from PIL import Image
import config
from utils import lmap
import copy

ImageFile.LOAD_TRUNCATED_IMAGES = True
Components = namedtuple('Components', ('N', 'Z', 'E'))

class SpectrogramBaseDataset(Dataset):
    """
    """
    __SEED = 448   # For randomly splitting the traintest set consistantly

    def __init__(self, img_path, divide_test, transform=None, test=False, resize=False, ignore=None,
        crop=False, crop_center=None, crop_padding=None, **kwargs):
        """

        :param img_path: path to the 'spectrograms' folder
        :param transform: list of transforms to apply AFTER resize and crop. Transforms must take a list of images as input
                          wrap with mytansforms.group_transforms if necessary
        :param test: boolean, training or test
        :param resize:  height, width or False
        :param crop:   height, width or False
        :param divide_test: 0 < divide_test < 1, percentage to divide test set into

        Order:
             - resize
             - crop
             - transform
        """
        super().__init__()

        ignore_names = ignore or []
        self.img_path = img_path
        self.test = test
        self.crop_padding = crop_padding

        # Transforms
        self.transform = transform

        if resize:
            resize = transforms.Resize(resize)

        self.resize = resize

        if crop and not crop_center:
            self.transform_random_same_crop = RandomSameCrop(crop)

        self.crop = crop
        self.crop_center = crop_center  
        
        # Get paths
        local_paths = self.get_spectrograms(img_path, 'local', ignore_names=ignore_names)
        noise_paths = self.get_spectrograms(img_path, 'noise', ignore_names=ignore_names)

        # Randomly shuffle the files the same way each time, to keep the test and train dataset the same
        self._shuffle(local_paths)
        self._shuffle(noise_paths)

        # Divide into training and test set
        test_local, train_local = self.separate_paths(local_paths, divide_test)
        test_noise, train_noise = self.separate_paths(noise_paths, divide_test)

        test_local, train_local, test_noise, train_noise = map(self.clean_paths, [test_local, train_local, test_noise, train_noise])

        if test:
            file_paths = test_local + test_noise
            self.local, self.noise = test_local, test_noise
        else:
            file_paths = train_local + train_noise
            self.local, self.noise = train_local, train_noise

        self.file_paths = self._shuffle(file_paths)

        self.labels = {
            0: 'noise',
            1: 'local',
        }

        self.reverse_map(self.labels)

    def separate_paths(self, paths, amount):
        if not (0 <= amount <= 1):
            raise ValueError("'amount' should be between 0 and 1")
        separate_index = int(len(paths) * amount)
        return paths[:separate_index], paths[separate_index:]

    def _shuffle(self, paths):
        random.seed(self.__SEED)
        random.shuffle(paths)
        return paths

    def shuffle(self):
        random.seed(self.__SEED)
        random.shuffle(self.file_paths)

    @property
    def num_classes(self):
        return len(self.labels) // 2   # Should always be divisble by 2 evenly

    @staticmethod
    def get_components(paths_to_components):
        return [glob.glob(os.path.join(path, "*.png")) for path in paths_to_components]


    def get_spectrograms(self, path, folder_name,  pattern='', ignore_names=None):
        """
        The base_loader allows the most flexibility in structuring your files, though it is probably overly complicated
        at the moment. Other loaders are designed simpler that inherit from base loader and modify this method.

        :param path: Base path containing folders that can contain events and noise.
                     Example:

                     /path/California
                     /path/Oklahoma
                     /path/Hawaii

        :param folder_name: Could be "noise" or "quakes" if following recommended file organization. Will return paths
                            within this folder. Example:

                            /path/California/noise
                            /path/California/quakes
                            ...

        :param pattern: Pattern to access spectrogram folders within the folder_name. The default setting assumes the
                        files set up as something like:

                            /path/California/quakes/1/first_component.png
                            /pathCalifornia/quakes/1/second_component.png
                            /path/California/quakes/1/vertical_component.png
                            ...

                        where the names of the components can be anything there should be 3 components per folder.


                        Custom patterns are passed as regex as a string and can allow specifying locations to the
                        spectrograms within the subfolders of the path:

                            /path/California/pattern/quakes/1/first_component.png

        :param ignore_names:
        :return:
        """
        folders = lmap(os.path.basename, glob.glob(os.path.join(path, '*')))
        folders = [f for f in folders if f not in ignore_names]

        def get_file_paths(folder_path):
            folders_path = os.path.join(path, folder_path, pattern, folder_name, '*/')
            subfolder_paths = glob.glob(folders_path)
            return self.get_components(subfolder_paths)
            
        file_paths = [] 
        for folder in folders:
            file_paths += get_file_paths(folder)

        # Maintain the same order each time, guaranteed with sorting

        file_paths.sort()
        return file_paths


    def apply_transforms(self, components):

        components = [self.open_image(component) for component in components]

        if self.resize:
            components = self.apply_resize(components)

        if self.crop:
            components = self.apply_crop(components)

        if self.transform:
            components = self.transform(components)

        n, z, e = components
        return n, z, e

    @staticmethod
    def open_image(img_path):
        img = Image.open(img_path)
        img = img.convert('RGB')
        return img

    def apply_crop(self, components):
        if self.crop_center:
            crop = transforms.CenterCrop(self.crop)
        else:
            img = components[0]
            self.transform_random_same_crop.set_params(img, self.crop, padding=self.crop_padding)
            crop = self.transform_random_same_crop
        n, z, e = map(crop, components)
        return n, z, e

    def apply_resize(self, components):
        n, z, e = map(self.resize, components)
        return n, z, e

    #@timing_msg("Getting item and applying transforms in data loader")
    def __getitem__(self, index, apply_transforms=True):
        n, z, e = self.file_paths[index]
        label = self.label_to_number(self.get_label(n))
        if apply_transforms:
            n, z, e = self.apply_transforms((n, z, e))
        return Components(n, z, e), label

    def get_labels(self):
        """ Returns a list of labels in order, corresponding to each sample in  self.file_paths """
        labels = []
        for i in range(len(self.file_paths)):
            n, z, e = self.file_paths[i]
            label = self.label_to_number(self.get_label(n))
            labels.append(label)
        return labels
        
    def _getitem_raw(self, index):
        """ Returns the components without any trasnforms applied """
        n, z, e = self.file_paths[index]
        label = self.label_to_number(self.get_label(n))
        n, z, e = [self.open_image(component) for component in (n, z, e)]
        n, z, e = map(transforms.ToTensor(), (n, z, e))
        return Components(n, z, e), label

    def get_next_index_with_label(self, label) -> int:
        if label not in self.labels:
            raise ValueError("Label is not in the labels: ", self.labels)

        for i, (sample, labl) in enumerate(self):
            if label == labl:
                yield i
        else:
            raise ValueError("No Label Found")

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

            # Sometimes need to convert to PIL Image, sometimes not dpeending on user settings
            # TODO: Be explicit and check for conditions to run the write code rather than try/except
            try:
                self.show_img(transforms.ToPILImage()(components[i]))
            except Exception as e:
                try:
                    self.show_img(components[i])
                except Exception as f:
                    raise

        if show:
            plt.show()

        return fig

    def preview_raw(self, index=0, show=True):
        components, label = self._getitem_raw(index)

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

        return fig

    @staticmethod
    def clean_paths(file_paths):
        return [components for components in file_paths if len(components) == 3]

    @staticmethod
    def get_label(file_path):
        return file_path.split('/')[-3]

    @staticmethod
    def reverse_map(dic):
        dic.update({v: k for k, v in dic.items()})

    @staticmethod
    def show_img(image):
        plt.imshow(image)
        plt.pause(0.001)
