from loaders.multiple_loader import SpectrogramMultipleDataset
import glob
import os
from utils import lmap

class SpectrogramDirectDataset(SpectrogramMultipleDataset):

    def __init__(self, path_pattern='', *args, **kwargs):  
        """
        :param pattern: str, the glob style pattern of how to get to the 'local' and 'noise' folders. 
        """
        self.pattern = path_pattern
        super().__init__(*args, **kwargs)

    def get_spectrograms(self, path, folder_name, *args, **kwargs):
        folder = os.path.join(path, folder_name)

        def get_components(paths_to_components):
            return [glob.glob(os.path.join(path, "*.png")) for path in paths_to_components]

        component_folders = glob.glob(os.path.join(folder, '*'))

        file_paths = get_components(component_folders) 

        # Maintain the same order each time, guaranteed with sorting
        file_paths.sort()

        return file_paths