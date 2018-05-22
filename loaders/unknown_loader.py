from loaders.base_loader import SpectrogramBaseDataset
import glob
import os

class SpectrogramUnknownDataset(SpectrogramBaseDataset):

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
        name = self.file_paths[index][0].split('/')[-2]
        return components, name