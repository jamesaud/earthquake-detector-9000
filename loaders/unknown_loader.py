from loaders.base_loader import SpectrogramBaseDataset
from loaders.direct_loader import SpectrogramDirectDataset
import glob
import os

class SpectrogramUnknownDatasetTimestamp(SpectrogramBaseDataset):
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

    def __getitem__(self, index, *args, **kwargs):
        """ Return the name of the item instead of the class
            The class is unkwown, because it is unseen data
        """
        components, label = super().__getitem__(index, *args, **kwargs)
        name = self.file_paths[index][0].split('/')[-2]                     # Gets the time from
        return components, name



class SpectrogramKnownDataset(SpectrogramDirectDataset):
    """
     Loads the names of the files instead of the label, because you don't know the label!
     This loader is when the name of the file is a timestamp.
     Use for if Noise and Events are in there own folders
     """


    def __getitem__(self, index, *args, **kwargs):
        """ Return the name of the item instead of the class
            The class is unkwown, because it is unseen data
        """
        components, label = super().__getitem__(index, *args, **kwargs)
        name = self.file_paths[index][0].rsplit('/', maxsplit=1)[0]  # Keep folder name but remove file name
        print(name)
        return components, name   # remove t