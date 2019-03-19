import torch.nn as nn
import torchvision.transforms as tf
import torch
import config
import mytransforms
from mytransforms import Group
from operator import mul

class mnist_three_component(nn.Module):


    DIM = 32               # 32
    NOISE_RGB_AMOUNT = 15  # 15  # Centered around this value, How much to change the value of a color [Guassian distribution added to a grayscale color value [0-255]
    BLUR = 2              # 2
    F_DIM = 64              # 64

    _transformations = [tf.Grayscale(num_output_channels=3),
                        tf.Resize((DIM, DIM)),
                        tf.ToTensor(),
                        #  mytransforms.NormalizeGray
                        ]

    _train = [
        tf.Grayscale(num_output_channels=1),
        mytransforms.Add1DNoise(config.BORDER_COLOR_GRAY, NOISE_RGB_AMOUNT),
        mytransforms.Gaussian_Blur(BLUR),
    ]

    _test = [
             ]

    transformations = {'train':  tf.Compose(_train + _transformations),
                       'test': tf.Compose(_test + _transformations)
                       }

    def __init__(self):
        F_DIM = self.F_DIM
        super().__init__()
        self.feats = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 5, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 5,  1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
        )

        self.classifier = nn.Sequential(
            nn.Linear(192*5*5, F_DIM, bias=False),
            nn.BatchNorm1d(F_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(F_DIM, F_DIM, bias=False),
            nn.BatchNorm1d(F_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(F_DIM, 2)
        )

    def forward(self, components):
        n, z, e = components
        nout, zout, eout = self.feats(n), self.feats(z), self.feats(e)
        out = torch.cat((nout, zout, eout), 1)
        out = out.view(-1, 192*5*5)
        out = self.classifier(out)
        return out


class mnist_three_component_exp(nn.Module):
    pass

class mnist_three_component_rgb(nn.Module):


    DIM = 32               # 32
    NOISE_RGB_AMOUNT = 10  # 15  # Centered around this value, How much to change the value of a color [Guassian distribution added to a grayscale color value [0-255]
    BLUR = 2            # 2
    F_DIM = 64              # 64

    _transformations = Group([tf.Resize((DIM, DIM)),
                              tf.ToTensor()])


    _train = [
            tf.RandomApply(Group([mytransforms.Add3DNoise(config.BORDER_COLOR_RGB, NOISE_RGB_AMOUNT)]), p=.5),
            tf.RandomApply(Group([mytransforms.Gaussian_Blur(BLUR)]), p=.5)
    ]


    _test = []

    transformations = {'train':  tf.Compose(_train + _transformations),
                       'test': tf.Compose(_test + _transformations)
                       }

    def __init__(self):
        F_DIM = self.F_DIM
        super().__init__()
        self.feats = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 5, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 5,  1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
        )

        self.classifier = nn.Sequential(
            nn.Linear(192*5*5, F_DIM, bias=False),
            nn.BatchNorm1d(F_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(F_DIM, F_DIM, bias=False),
            nn.BatchNorm1d(F_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(F_DIM, 2)
        )

    def forward(self, components):
        n, z, e = components
        nout, zout, eout = self.feats(n), self.feats(z), self.feats(e)
        out = torch.cat((nout, zout, eout), 1)
        out = out.view(-1, 192*5*5)
        out = self.classifier(out)
        return out