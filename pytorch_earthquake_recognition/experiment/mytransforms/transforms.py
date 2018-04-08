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

def Add1DNoise(IGNORE_COLOR, NOISE_RGB_AMOUNT):
    return transforms.Lambda(lambda img: _add_noise(img, IGNORE_COLOR, NOISE_RGB_AMOUNT))

def Gaussian_Blur(radius=2):
    return transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=radius)))


NormalizeGray = transforms.Normalize(mean=config.GRAY_MEAN,
                                     std=config.GRAY_STD)

NormalizeColor = transforms.Normalize(mean=config.RGB_MEAN,
                                      std=config.RGB_STD)
