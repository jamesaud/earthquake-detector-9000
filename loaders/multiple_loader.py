from loaders.base_loader import SpectrogramBaseDataset

class SpectrogramMultipleDataset(SpectrogramBaseDataset):
    """
    """

    def __init__(self, img_path, divide_test, transform=None, test=False, **kwargs):   # 30%
        super().__init__(img_path, divide_test, transform, test, **kwargs)
