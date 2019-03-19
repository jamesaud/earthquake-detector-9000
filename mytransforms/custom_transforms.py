import torchvision.transforms.functional as F
from torchvision import transforms
import random
from functools import wraps
from typing import List

__all__ = ['RandomSameCrop', 'RandomSameCropWidth', 'transform_group', 'Group']

class RandomSameCrop(transforms.RandomCrop):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def set_params(self, img, size, padding=0):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop in (height, width).

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        self.padding = padding
        self.params = super().get_params(img, size)

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """

        i, j, h, w = self.params

        return F.crop(img, i, j, h, w)

class RandomSameCropWidth(transforms.RandomCrop):
    """
    Performs the SAME random crop on multiple images.
    call set_params to change the random crop values.
    """

    def __init__(self, *args, padding=None, **kwargs):
        super().__init__(*args ,**kwargs)
        self.params = None
        
    def get_params(self, img, output_width):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = h, output_width
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(self.padding_left, w - tw - self.padding_right)
        return i, j, th, tw

    def set_params(self, img, size, padding=0):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        output_width = size[1]
        self.padding_left = padding[0] if padding else 0 
        self.padding_right = padding[1] if padding else 0
        self.params = self.get_params(img, output_width)

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """

        i, j, h, w = self.params

        return F.crop(img, i, j, h, w)



def transform_group(transform):
    """ 
    A wrapper that takes a normal transform, and allows it to take a list of images as the argument
    The transform will then apply to all the images.
    This allows normal transforms to be composed with custom ones that take multiple images as input
    """
    @wraps(transform)
    def apply_transform_to_group(imgs, *args, **kwargs):
        imgs = list(map(transform, imgs))
        return imgs

    return apply_transform_to_group


def Group(transforms: List):
    """ Applies transform_group to a list of transforms"""
    return [transform_group(transform) for transform in transforms]


class ComposeGroup(transforms.Compose):
    """
    Same as transform group, but takes a list of normal transforms and applies them all together
    """
    def __call__(self, imgs):
        for t in self.transforms:
            imgs = map(t, imgs)
        return list(imgs)

