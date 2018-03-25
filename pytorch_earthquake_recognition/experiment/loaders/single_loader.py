from loaders.base_loader import SpectrogramBaseDataset
from matplotlib import pyplot as plt
from torchvision import transforms

class SpectrogramSingleDataset(SpectrogramBaseDataset):
    """
    """
    __SEED = 448

    RESIZE_WIDTH = 310
    RESIZE_HEIGHT = 200

    CROP_WIDTH = 260
    CROP_HEIGHT = 200

    def __init__(self, img_path, transform=None, test=False, divide_test=.3):   # 30%
        super().__init__(img_path, transform, test, divide_test=divide_test)


    def __getitem__(self, index):
        components, label = super().__getitem__(index)
        return components.Z, label

    def preview(self, index=0, show=True):
        image, label = self.__getitem__(index)

        fig = plt.figure()
        plt.suptitle(self.labels[label])
        ax = plt.subplot(1, 1, 1)
        plt.tight_layout()
        ax.set_title("Z Component")
        ax.axis('off')
        self.show_img(transforms.ToPILImage()(image))
        if show:
            plt.show()


if __name__ == '__main__':
    IMG_PATH = '../spectrograms'
    s = SpectrogramSingleDataset(IMG_PATH)
    s.preview()
    s = iter(s)
    print(next(s))