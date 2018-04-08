import torch.nn as nn
import torchvision.transforms as transforms
from mytransforms.transforms import NormalizeGray
import torch.nn.functional as F
import torch

class mnist_simple(nn.Module):
    DIM = 16               # by 16
    BORDER_COLOR = 30      # Ignore border color when adding NOISE
    NOISE_RGB_AMOUNT = 4   # How much to change the value of a color [Guassian distribution added to a grayscale color value [0-255]

    _transformations = [transforms.Grayscale(num_output_channels=3),
                         transforms.Resize((DIM, DIM)),  # 16 x 16
                         transforms.ToTensor(),
                         NormalizeGray
                         ]

    _train = [#transforms.Grayscale(num_output_channels=1),
               #mytransforms.Add1DNoise(BORDER_COLOR, NOISE_RGB_AMOUNT),
               ]

    _test = []

    transformations = {'train': transforms.Compose(_train + _transformations),
                       'test': transforms.Compose(_test + _transformations)
                       }

    def __init__(self):
        super().__init__()
        self.dim_feats = 3*self.DIM*2*2*2

        self.feats = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
        )

        self.classifier = nn.Sequential(nn.Linear(self.dim_feats, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5),
                                        nn.Linear(128, 2)
                                        )

    def forward(self, components):
        n, z, e = components
        nout, zout, eout = self.feats(n), self.feats(z), self.feats(e)

        out = torch.cat((nout, zout, eout), 1)
        out = out.view(-1, self.dim_feats)
        out = self.classifier(out)
        return F.log_softmax(out, dim=1)


