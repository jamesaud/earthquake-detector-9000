import torchvision.transforms.functional as F
from torchvision import transforms
import random


__all__ = ['RandomSameCrop', 'RandomSameCropWidth', 'RandomApplyConsistent']

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


class RandomApplyConsistent(transforms.RandomApply):
    """
    Apply randomly a list of transformations with a given probability

    Args:
        transforms (list or tuple): list of transformations
        p (float): probability

    Applies randomly consistently, so if it was
    """

    def __init__(self, transforms, p=0.5):
        super().__init__(transforms)
        self.p = p
        self.random_val = self.random_val()

    def __call__(self, img):
        if self.p < self.random_val:
            return img
        for t in self.transforms:
            img = t(img)
        return img

    def reset(self):
        self.random_val = self.random_val()

    def random_val(self):
        return random.random()