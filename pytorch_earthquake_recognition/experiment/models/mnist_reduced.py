import torch.nn as nn
import torchvision.transforms as transforms


class mnist_reduced(nn.Module):
    __transformations = [transforms.Grayscale(num_output_channels=3),
                         transforms.Resize((16, 16)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.0007967819185817943, 0.0007967819185817943, 0.0007967819185817943],
                                              std=[0.0002987987562721851, 0.0002987987562721851, 0.0002987987562721851])]

    __train = [transforms.Resize((200, 310))]

    __test = [transforms.Resize((200, 310))]

    transformations = {'train':  transforms.Compose(__train + __transformations),
                       'test': transforms.Compose(__test + __transformations)
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

    def forward(self, inputs):
        out = self.feats(inputs)
        out = self.dropout(out)
        out = self.classifier(out)
        out = self.avgpool(out)
        out = out.view(-1, 3)
        return out




