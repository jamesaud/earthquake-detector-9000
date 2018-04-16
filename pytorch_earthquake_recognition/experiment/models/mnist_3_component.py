import torch.nn as nn
import torchvision.transforms as transforms
import torch
import config
from mytransforms import transforms as mytransforms


class mnist_three_component(nn.Module):


    DIM = 64
    NOISE_RGB_AMOUNT = 3  # How much to change the value of a color [Guassian distribution added to a grayscale color value [0-255]

    _transformations = [transforms.Grayscale(num_output_channels=3),
                         transforms.Resize((DIM, DIM)),
                         transforms.ToTensor(),
                         mytransforms.NormalizeGray
                         ]

    _train = [
        # transforms.Grayscale(num_output_channels=1),
        # mytransforms.Add1DNoise(config.BORDER_COLOR_GRAY, NOISE_RGB_AMOUNT),
        # mytransforms.Gaussian_Blur(2),
    ]

    _test = []

    transformations = {'train':  transforms.Compose(_train + _transformations),
                       'test': transforms.Compose(_test + _transformations)
                       }

    def __init__(self):
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
            nn.Linear(192*13*13, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(128, 2)
        )

    def forward(self, components):
        n, z, e = components
        nout, zout, eout = self.feats(n), self.feats(z), self.feats(e)
        out = torch.cat((nout, zout, eout), 1)
        out = out.view(-1, 192*13*13)
        out = self.classifier(out)
        return out