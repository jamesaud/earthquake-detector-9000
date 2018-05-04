import torch.nn as nn
import torchvision.transforms as transforms
import torch
import config
from mytransforms import transforms as mytransforms


class feed_forward(nn.Module):


    DIM = 32
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

        self.classifier = nn.Sequential(
            nn.Linear(9*32*32, 10),
            nn.ReLU(inplace=True),

            nn.Linear(10, 30),
            nn.ReLU(inplace=True),

            nn.Linear(30, 10),
            nn.ReLU(inplace=True),

            nn.Linear(10, 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, components):
        n, z, e = components
        out = torch.cat((n, z, e), 1)
        out = out.view(-1, 9 * 32 * 32)
        out = self.classifier(out)
        return out