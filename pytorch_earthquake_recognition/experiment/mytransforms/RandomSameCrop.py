import torchvision.transforms.functional as F
from torchvision import transforms
import random

class RandomSameCropWidth(transforms.RandomCrop):
    """
    Performs the SAME random crop on multiple images.
    call set_params to change the random crop values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args ,**kwargs)
        self.params = None

    @staticmethod
    def get_params(img, output_width):
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
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def set_params(self, img, output_width):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        self.params = self.get_params(img, output_width)

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            img = F.pad(img, self.padding)

        i, j, h, w = self.params

        return F.crop(img, i, j, h, w)
