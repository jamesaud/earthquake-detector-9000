from loaders.multiple_loader import SpectrogramMultipleDataset
import glob
import os
from utils import lmap
from pathlib import Path


class SpectrogramDirectDataset(SpectrogramMultipleDataset):

    def __init__(self, path_pattern='', *args, **kwargs):  
        """
        :param pattern: str, the glob style pattern of how to get to the 'local' and 'noise' folders. 
        """
        self.pattern = path_pattern
        super().__init__(*args, **kwargs)

    def get_spectrograms(self, path, folder_name, *args, **kwargs):

        folder = os.path.join(path, folder_name)

        def get_components(path_to_components):
            return [str(component) for component in Path(path_to_components).iterdir() if component.suffix == '.png']

        component_folders = glob.glob(os.path.join(folder, '*'))

        ## ADDED
        file_paths = list(map(get_components, component_folders))
        file_paths.sort()  # Maintain the same order each time, guaranteed with sorting
        return file_paths

