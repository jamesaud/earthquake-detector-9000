from loaders.base_loader import SpectrogramBaseDataset
from matplotlib import pyplot as plt
from torchvision import transforms

class SpectrogramSingleDataset(SpectrogramBaseDataset):
    """
    """

    def __init__(self, img_path, divide_test, transform=None, test=False, **kwargs):   # 30%
        super().__init__(img_path, divide_test, transform, test, **kwargs)

    def __getitem__(self, index):
        components, label = super().__getitem__(index)
        return components.Z, label

    def _getitem_raw(self, index):
        components, label = super()._getitem_raw(index)
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

        return fig


    def preview_raw(self, index=0, show=True):
        image, label = self._getitem_raw(index)

        fig = plt.figure()
        plt.suptitle(self.labels[label])
        ax = plt.subplot(1, 1, 1)
        plt.tight_layout()
        ax.set_title("Z Component")
        ax.axis('off')
        self.show_img(transforms.ToPILImage()(image))

        if show:
            plt.show()

        return fig


if __name__ == '__main__':
    IMG_PATH = '../spectrograms'
    s = SpectrogramSingleDataset(IMG_PATH)
    s.preview()
    s = iter(s)
    print(next(s))