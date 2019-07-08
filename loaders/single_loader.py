from loaders.base_loader import SpectrogramBaseDataset
from loaders.direct_loader import SpectrogramDirectDataset
from matplotlib import pyplot as plt
from torchvision import transforms

class SpectrogramSingleDataset(SpectrogramDirectDataset):
    """
    """
    component_index = {"N": 0, "Z": 1, "E": 2}
    def __init__(self, component='E', *args, **kwargs):   # 30%
        if component not in self.component_index:
            raise ValueError(f"Component should be one of {component_index.keys()}")
        self.component = component
        super().__init__(*args, **kwargs)

    @property
    def index(self):
        return self.component_index[self.component]

    def __getitem__(self, *args, **kwargs):
        components, label = super().__getitem__(*args, **kwargs)
        return components[self.index], label

    def _getitem_raw(self, *args, **kwargs):
        components, label = super()._getitem_raw(*args, **kwargs)
        return components[self.index], label

    def preview(self, index=0, show=True):
        image, label = self.__getitem__(index)

        fig = plt.figure()
        plt.suptitle(self.labels[label])
        ax = plt.subplot(1, 1, 1)
        plt.tight_layout()
        ax.set_title(self.component + " Component")
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
        ax.set_title(self.component + " Component")
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