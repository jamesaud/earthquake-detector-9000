import torch.nn as nn
import torchvision.transforms as tf
import torch
import config
import mytransforms
from mytransforms import Group, transform_group 

class mnist_three_component(nn.Module):

    NOISE_RGB_AMOUNT = 15  # 15
    BLUR = 2               # 2
    DIM = 32
    F_DIM = 64             # 64

    _transformations = Group([tf.Grayscale(num_output_channels=3),   # Need 3 channels for
                              tf.Resize((DIM, DIM)),
                              tf.ToTensor(),
  #                            mytransforms.NormalizeGray
                        ])
 
    # Random Apply should not be in Group, so that the same random apply is applied to all 3 images
    _train = [transform_group(tf.Grayscale(num_output_channels=1)), 
      #        tf.RandomApply(Group([mytransforms.Add3DNoise(config.BORDER_COLOR_RGB, NOISE_RGB_AMOUNT)]), p=.5),
      #        tf.RandomApply(Group([mytransforms.Gaussian_Blur(BLUR)]), p=.5)
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
            nn.Linear(4800, F_DIM, bias=False),
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
        #print(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class mnist_three_component_exp(nn.Module):
    DIM = 32               # 32
    NOISE_RGB_AMOUNT = 3   # 15 
    BLUR = 2               # 2
    F_DIM = 32             # 64

    _transformations = Group([tf.Grayscale(num_output_channels=3),   # Need 3 channels for
                              tf.Resize((DIM, DIM)),
                              tf.ToTensor(),
                              mytransforms.NormalizeGray
                        ])
 
    # Random Apply should not be in Group, so that the same random apply is applied to all 3 images
    _train = [ #transform_group(tf.Grayscale(num_output_channels=1)), 
        #      tf.RandomApply(Group([mytransforms.Add3DNoise(config.BORDER_COLOR_RGB, NOISE_RGB_AMOUNT)]), p=.5),
        #      tf.RandomApply(Group([mytransforms.Gaussian_Blur(BLUR)]), p=.5)
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
            nn.Linear(4800, F_DIM, bias=False),
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
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class mnist_three_component_rgb(nn.Module):

    DIM = 64               # 32
    NOISE_RGB_AMOUNT = 3  # 15 
    BLUR = 1               # 2
    F_DIM = 128              # 64

    _transformations = Group([tf.Resize((DIM, DIM)),
                              tf.ToTensor(),
                              mytransforms.NormalizeColor])

    _train = [
            #   tf.RandomApply(Group([mytransforms.Add3DNoise(config.BORDER_COLOR_RGB, NOISE_RGB_AMOUNT)]), p=.5),
            #   tf.RandomApply(Group([mytransforms.Gaussian_Blur(BLUR)]), p=.5)
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
            nn.Linear(32448, F_DIM, bias=False),
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
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


