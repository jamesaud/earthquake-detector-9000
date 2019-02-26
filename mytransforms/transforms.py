from torchvision import transforms
import PIL
import numpy as np
from random import getrandbits
from PIL import ImageFilter
import config

def _print_state(img):
    print(img)
    return img


def _add_noise(img, BORDER_COLOR, NOISE_RGB_AMOUNT):
    """Add noise to a grayscaled 1 channel image"""
    # Random boolean
    if getrandbits(1):
        img = np.array(img)
        condition = img != BORDER_COLOR
        noise = np.random.normal(0, NOISE_RGB_AMOUNT, size=img[condition].shape).astype(int)
        img[condition] = noise + img[condition]

        # Convert array to Image
        img = PIL.Image.fromarray(img)

    return img

def PrintState():
    return transforms.Lambda(lambda img: _print_state(img))


class Lambda(transforms.Lambda):
    """ Add docstring to transforms.Lambda """
    def __init__(self, lambd, description=""):
        super().__init__(lambd)
        self.description = description

    def __repr__(self):
        return self.__class__.__name__ + '(): ' + self.description


def Add1DNoise(IGNORE_COLOR, NOISE_RGB_AMOUNT):
    """ Adds noise to every pixel except the 'ignore_color'. useful to not add noise to borders """
    return Lambda(lambda img: _add_noise(img, IGNORE_COLOR, NOISE_RGB_AMOUNT),
                  f"Add1DNoise(IGNORE_COLOR={IGNORE_COLOR}, NOISE_RGB_AMOUNT={NOISE_RGB_AMOUNT})")

def Gaussian_Blur(radius):
    return Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=radius)),
                  f"Gaussian_Blur(radius={radius})")


NormalizeGray = transforms.Normalize(mean=config.GRAY_MEAN,
                                     std=config.GRAY_STD)

NormalizeColor = transforms.Normalize(mean=config.RGB_MEAN,
                                      std=config.RGB_STD)
