import os
from unittest import TestCase
from loaders.base_loader import SpectrogramBaseDataset
from loaders.custom_path_loader import SpectrogramCustomPathDataset
from torchvision import transforms
import re

class TestLoader(TestCase):
    IMG_PATH = 'tests/spectrograms'
    WIDTH = 316
    HEIGHT = 217

    def setUp(self):
       self.dataset = SpectrogramBaseDataset(img_path='', divide_test=.2, transform=None)
       self.path = os.path.join(os.getcwd(), self.IMG_PATH) 

    def test_get_spectrograms(self):
        ignore = 'AlaskaQuakes'
        paths = self.dataset.get_spectrograms(self.path, 'noise', pattern='', ignore_names=[ignore])
        self.assertGreater(len(paths), 10)
        self.assertFalse([path for path in paths if ignore in path])


    def test_cropping(self):
        TRANSFORMS = transforms.Compose(
            [transforms.ToTensor()]
        )
        crop = (200, 200)
        crop_padding = (50, 0, 0, 0)

        def get_components(dataset):
            idtrain = iter(dataset)
            components, label = next(idtrain)
            return components

        def assert_components_correct(components, width, height):
            for component in components:
                _, h, w = component.size()
                self.assertEqual(h, height)
                self.assertEqual(w, width)

        args = dict(img_path=self.IMG_PATH, divide_test=0, transform=TRANSFORMS, crop=crop)
        components = get_components(SpectrogramCustomPathDataset(crop_padding=crop_padding, **args))
        assert_components_correct(components, 200, 200)

        components = get_components(SpectrogramCustomPathDataset(crop_center=True, **args))

        assert_components_correct(components, 200, 200)


    