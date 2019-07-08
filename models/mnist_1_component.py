import torch.nn as nn
import torchvision.transforms as transforms
import mytransforms.transforms as mytransforms
from mytransforms import Group, transform_group 


class mnist_one_component(nn.Module):
    _transformations = Group([
                         transforms.Grayscale(num_output_channels=3),
                         transforms.Resize((32, 32)),
                         transforms.ToTensor(),
                         mytransforms.NormalizeGray
                        ])

    _train = []

    _test = []

    transformations = {'train':  transforms.Compose(_train + _transformations),
                       'test': transforms.Compose(_test + _transformations)
                       }

    def __init__(self):
        super().__init__()
        self.feats = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 5, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3,  1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

        )

        self.classifier = nn.Conv2d(64, 3, 1)
        self.avgpool = nn.AvgPool2d(3, 3)
        self.dropout = nn.Dropout(0.5)

        self.linear = nn.Sequential(
            nn.Linear(3*2*2, 128),
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
        inputs = components
        out = self.feats(inputs)
        out = self.dropout(out)
        out = self.classifier(out)
        out = self.avgpool(out)
        out = out.view(-1, 3 * 2 * 2)
        out = self.linear(out)
        return out



