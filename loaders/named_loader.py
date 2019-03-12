from loaders.base_loader import SpectrogramBaseDataset
from loaders.direct_loader import SpectrogramDirectDataset
import glob
import os
from abc import ABC, abstractmethod


class AbstractSpectrogramNamedDataset(ABC, SpectrogramBaseDataset):

    def __init__(self, return_name=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_name = return_name

    @abstractmethod
    def name_from_filepath(self, path: str):
        pass

    def __getitem__(self, index, *args, **kwargs):
        """ Return the name of the item instead of the class
            The class is unkwown, because it is unseen data
        """
        components, label = super().__getitem__(index, *args, **kwargs)
        filename = self.file_paths[index][0]
        name = self.name_from_filepath(filename)

        if self.return_name:
            return components, label, name
        return components, label  # The true label is unknown for this dataset!


class SpectrogramNamedTimestampDataset(AbstractSpectrogramNamedDataset):
    """
     Loads the names of the files instead of the label, because you don't know the label!
     This loader is when the name of the file is a timestamp.
     Should be a folder all continuous spectrogrr own folder
     """

    def get_spectrograms(self, path, *args, **kwargs):
        file_paths = glob.glob(os.path.join(path, '*'))
        paths = self.get_components(file_paths)
        paths.sort()
        return paths

    def label_to_number(self, path):
        return 0

    def name_from_filepath(self, path):
        return path.split('/')[-2]


class SpectrogramNamedDataset(AbstractSpectrogramNamedDataset, SpectrogramDirectDataset):
    """
     Loads the names of the files instead of the label, because you don't know the label!
     This loader is when the name of the file is a timestamp.
     Use for if Noise and Events are in there own folders
     """

    def name_from_filepath(self, path):
        return path.rsplit('/', maxsplit=1)[0]

