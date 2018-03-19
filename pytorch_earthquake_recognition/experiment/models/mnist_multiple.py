import torch.nn as nn
import torchvision.transforms as transforms
import torch
import numpy as np
from random import getrandbits
import PIL
import mytransforms.transforms as mytransforms

class mnist_multiple_1(nn.Module):
    __transformations = [transforms.Grayscale(num_output_channels=3),
                         transforms.Resize((32, 32)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.0007967819185817943, 0.0007967819185817943, 0.0007967819185817943],
                                              std=[0.0002987987562721851, 0.0002987987562721851, 0.0002987987562721851])]

    __train = [transforms.Resize((200, 310)),
               #transforms.RandomCrop((200, 250))
               ]

    __test = [transforms.Resize((200, 310)),
              #transforms.RandomCrop((200, 250))
              ]

    transformations = {'train':  transforms.Compose(__train + __transformations),
                       'test': transforms.Compose(__test + __transformations)
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

            nn.Conv2d(64, 64, 3,  1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
        )

        self.classifier = nn.Conv2d(192, 3, 1)

        self.avgpool = nn.AvgPool2d(6, 6)
        self.dropout = nn.Dropout(0.5)


    def forward(self, components):
        n, z, e = components
        nout, zout, eout = self.feats(n), self.feats(z), self.feats(e)
        out = torch.cat((nout, zout, eout), 1)
        out = self.dropout(out)
        out = self.classifier(out)
        out = self.avgpool(out)
        out = out.view(-1, 3)
        return out


class mnist_multiple_2(nn.Module):
    __transformations = [transforms.Grayscale(num_output_channels=3),
                         transforms.Resize((32, 32)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.0007967819185817943, 0.0007967819185817943, 0.0007967819185817943],
                                              std=[0.0002987987562721851, 0.0002987987562721851, 0.0002987987562721851])]

    __train = [transforms.Resize((200, 310)),
               transforms.RandomCrop((200, 250))
               ]

    __test = [transforms.Resize((200, 310)),
              transforms.RandomCrop((200, 250))
              ]

    transformations = {'train':  transforms.Compose(__train + __transformations),
                       'test': transforms.Compose(__test + __transformations)
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

            nn.Conv2d(64, 64, 3,  1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
        )

        self.classifier = nn.Sequential(
            nn.Linear(192 * 7 * 7, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 3)
        )

    def forward(self, components):
        n, z, e = components
        nout, zout, eout = self.feats(n), self.feats(z), self.feats(e)
        out = torch.cat((nout, zout, eout), 1)
        out = out.view(-1, 192 * 7 * 7)
        out = self.classifier(out)
        return out


class mnist_multiple(nn.Module):

    DIM = 24  # by 16
    BORDER_COLOR = 30
    NOISE_RGB_AMOUNT = 4

    __transformations = [transforms.Grayscale(num_output_channels=3),
                         transforms.Resize((DIM, DIM)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.0007967819185817943, 0.0007967819185817943, 0.0007967819185817943],
                                              std=[0.0002987987562721851, 0.0002987987562721851, 0.0002987987562721851])
                         ]

    __train = [transforms.Grayscale(num_output_channels=1),
               mytransforms.Add1DNoise(BORDER_COLOR, NOISE_RGB_AMOUNT),
               ]

    __test = []

    transformations = {'train':  transforms.Compose(__train + __transformations),
                       'test': transforms.Compose(__test + __transformations)
                       }

    def __init__(self):
        super().__init__()
        self.feats = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(8),

            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, 5, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, 3,  1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
        )

        self.classifier = nn.Sequential(
            nn.Linear((self.DIM * 3) * 3 * 3, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, 3)
        )

    def forward(self, components):
        n, z, e = components
        nout, zout, eout = self.feats(n), self.feats(z), self.feats(e)
        out = torch.cat((nout, zout, eout), 1)
        out = out.view(-1, (self.DIM * 3) * 3 * 3)
        out = self.classifier(out)
        return out
