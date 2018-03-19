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

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_PATH = '../spectrograms'
IMG_EXT = '.png'

Components = namedtuple('Components', ('N', 'Z', 'E'))


class Spectrogram3ComponentDataset(Dataset):
    """
    """
    __SEED = 448

    RESIZE_WIDTH = 310
    RESIZE_HEIGHT = 200

    CROP_WIDTH = 260
    CROP_HEIGHT = 200

    def __init__(self, img_path, transform=None, test=False, random_crop=True, divide_test=30):   # 30%
        self.img_path = img_path
        self.random_crop = random_crop

        # Transforms
        self.transform = transform
        self.resize = transforms.Resize((self.RESIZE_HEIGHT, self.RESIZE_WIDTH))
        self.transform_random_same_crop = RandomSameCropWidth(self.CROP_WIDTH)

        path_to_components = os.path.join(img_path + '/*/*/*/')
        folder_paths = glob.glob(path_to_components)
        image_paths = [glob.glob(os.path.join(folder, "*.png")) for folder in folder_paths]

        # Randomly shuffle the files the same way each time, to keep the test and train dataset the same
        random.seed(self.__SEED)
        random.shuffle(image_paths)

        separate_index = int(len(image_paths) * (divide_test / 100))
        test_paths, train_paths = image_paths[:separate_index], image_paths[separate_index:]

        self.file_paths = test_paths if test else train_paths

        self.file_paths = self.clean_paths(self.file_paths)

        self.labels = {
            'noise': 0,
            'local': 1,
            'non_local': 2
        }

        self.reverse_map(self.labels)


    def __getitem__(self, index):
        n, z, e = self.file_paths[index]

        # The same random crop should be set for all 3 components
        random_crop_set = False

        def path_to_img(path):
            nonlocal random_crop_set

            img = Image.open(path)
            img = img.convert('RGB')

            if self.transform:
                img = self.resize(img)

                # Apply
                if self.random_crop and (not random_crop_set):
                    self.transform_random_same_crop.set_params(img, self.CROP_WIDTH)
                    random_crop_set = True

                img = self.transform_random_same_crop(img)
                img = self.transform(img)

            return img

        label = self.label_to_number(self.get_label(n))
        n, z, e = map(path_to_img, [n, z, e])
        return Components(n, z, e), label

    def __len__(self):
        return len(self.file_paths)

    def label_to_number(self, label):
        return self.labels[label]

    def preview(self, index=0):
        components, label = self.__getitem__(index)

        fig = plt.figure()
        plt.suptitle(self.labels[label])
        for i, title in enumerate(['n', 'z', 'e']):
            ax = plt.subplot(1, 3, i + 1)
            plt.tight_layout()
            ax.set_title(title)
            ax.axis('off')
            self.show_img(transforms.ToPILImage()(components[i]))

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
    dataset_train = Spectrogram3ComponentDataset(IMG_PATH)
    dt = iter(dataset_train)
    imgs, label = next(dt)

    dataset_train.preview()
