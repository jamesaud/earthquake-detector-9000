import torch.nn as nn
import torchvision.transforms as transforms
import torch
import mytransforms.transforms as mytransforms
import config

class mnist_multiple(nn.Module):

    DIM = 128  # by 32
    NOISE_RGB_AMOUNT = 8  # How much to change the value of a color [Guassian distribution added to a grayscale color value [0-255]

    _transformations = [transforms.Grayscale(num_output_channels=3),
                         transforms.Resize((DIM, DIM)),
                         transforms.ToTensor(),
                         mytransforms.NormalizeGray
                         ]

    _train = [
               transforms.Grayscale(num_output_channels=1),
               mytransforms.Add1DNoise(config.BORDER_COLOR_GRAY, NOISE_RGB_AMOUNT),
               mytransforms.Gaussian_Blur(2),
               ]

    _test = []

    transformations = {'train':  transforms.Compose(_train + _transformations),
                       'test': transforms.Compose(_test + _transformations)
                       }

    def __init__(self):
        super().__init__()
        self.dim = 3 * 64 * 13 * 13

        self.feats1 = self.make_features()
        self.feats2 = self.make_features()
        self.feats3 = self.make_features()

        self.classifier = nn.Sequential(
            nn.Linear(self.dim, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )


    @staticmethod
    def make_features():
        return nn.Sequential(
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

    def apply_feats(self, n, z, e):
        return self.feats1(n), self.feats2(z), self.feats3(e)

    def add_dimensions(self, n, z, e):
        return n.unsqueeze(1), z.unsqueeze(1), e.unsqueeze(1)

    def forward(self, components):
        n, z, e = components
        n, z, e = self.apply_feats(n, z, e)
        n, z, e = self.add_dimensions(n, z, e)
        out = torch.cat((n, z, e), 1)
        print(out)
        out = out.view(out, self.dim)
        out = self.classifier(out)
        return out
