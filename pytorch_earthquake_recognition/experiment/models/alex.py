import torch.nn as nn
import torchvision.transforms as transforms



class AlexNet(nn.Module):

    normalize = transforms.Normalize(mean=[0.0007967819185817943, 0.0007967819185817943, 0.0007967819185817943],
                                     std=[0.0002987987562721851, 0.0002987987562721851, 0.0002987987562721851])

    grayscale = transforms.Grayscale(num_output_channels=3)

    __train = [transforms.Resize((200, 310)),
               transforms.RandomCrop((200, 250))]

    __test = [transforms.Resize((200, 310)),
              transforms.RandomCrop((200, 250))]

    transformations = {
        'train': transforms.Compose(__train + [
            grayscale,
            transforms.Resize((224, 224)),  # 256
            transforms.ToTensor(),
            normalize
        ]),
        'test': transforms.Compose(__test + [
            grayscale,
            transforms.Resize((224, 224)),   # 256
            transforms.ToTensor(),
            normalize
        ])
    }

    def __init__(self, num_classes=3):
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
            nn.Linear(256 * 6 * 6, 4096, bias=False),
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
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x




import torch

class AlexNetMultiple(nn.Module):

    normalize = transforms.Normalize(mean=[0.0007967819185817943, 0.0007967819185817943, 0.0007967819185817943],
                                     std=[0.0002987987562721851, 0.0002987987562721851, 0.0002987987562721851])

    grayscale = transforms.Grayscale(num_output_channels=3)

    __train = [transforms.Resize((200, 310)),
               transforms.RandomCrop((200, 260))]

    __test = [transforms.Resize((200, 310)),
              transforms.RandomCrop((200, 260))]

    transformations = {
        'train': transforms.Compose(__train + [
            grayscale,
            transforms.Resize((224, 224)),  # 256
            transforms.ToTensor(),
            normalize
        ]),
        'test': transforms.Compose(__test + [
            grayscale,
            transforms.Resize((224, 224)),   # 256
            transforms.ToTensor(),
            normalize
        ])
    }

    def __init__(self, num_classes=3):
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
            nn.Linear(256 * 6 * 6 * 3, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )


    def forward(self, components):
        n, z, e = components
        nout, zout, eout = self.features(n), self.features(z), self.features(e)
        x = torch.cat((nout, zout, eout), 0)
        x = x.view(-1, 256 * 6 * 6 * 3)
        x = self.classifier(x)
        return x


