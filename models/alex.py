import torch.nn as nn
import torchvision.transforms as transforms
import mytransforms.transforms as mytransforms
from mytransforms import Group
import torch
import config

class AlexNet(nn.Module):

    normalize = mytransforms.NormalizeGray
    grayscale = transforms.Grayscale(num_output_channels=3)


    _transformations = Group([
            transforms.ToTensor(),
        ])

    _train = []

    _test = []

    transformations = {
        'train': transforms.Compose(_train + _transformations),
        'test': transforms.Compose(_test + _transformations)
    }

    def __init__(self, num_classes=2):
        super().__init__()
        self.DIM = 256 * 5 * 8

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),

            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),

            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.DIM, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.DIM)
        x = self.classifier(x)
        return x



class AlexNetMultiple(nn.Module):

    normalize = transforms.Normalize(mean=config.RGB_MEAN,
                                     std=config.RGB_STD)

    _train = []
    _test = []
    _transformations = Group([transforms.ToTensor()])

    transformations = {
        'train': transforms.Compose(_train + _transformations),
        'test':  transforms.Compose(_test + _transformations)
    }

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2,
                      bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),

            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),

            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
        self.classifier = nn.Sequential(
            nn.Linear(768*4*7, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )


    def forward(self, components):
        n, z, e = components
        nout, zout, eout = self.features(n), self.features(z), self.features(e)
        x = torch.cat((nout, zout, eout), 1)
        x = x.view(-1, 768*4*7)
        x = self.classifier(x)
        return x


