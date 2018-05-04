from loaders.multiple_loader import SpectrogramMultipleDataset

class SpectrogramCustomPathDataset(SpectrogramMultipleDataset):
    """
    """

    def __init__(self, path_pattern='', *args, **kwargs):
        """
        :param pattern: str, the glob style pattern of how to get to the 'local' and 'noise' folders. 
        """
        self.pattern = path_pattern
        super().__init__(*args, **kwargs)

    def get_spectrograms(self, *args, **kwargs):
        return super().get_spectrograms(*args, pattern=self.pattern, **kwargs)


