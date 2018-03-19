from torchvision.models.vgg import make_layers, VGG, cfg, model_zoo, model_urls
from functools import partial
import torchvision.transforms as transforms

class MyVGG(VGG):
    __transformations = [transforms.Grayscale(num_output_channels=3),
                         transforms.ToTensor(),
                         transforms.Normalize(
                             mean=[0.0007967819185817943, 0.0007967819185817943, 0.0007967819185817943],
                             std=[0.0002987987562721851, 0.0002987987562721851, 0.0002987987562721851])]

    __train = [transforms.Resize((200, 310)),
               transforms.RandomCrop((200, 270))
               ]

    __test = [transforms.Resize((200, 310)),
              transforms.RandomCrop((200, 270))
              ]

    transformations = {'train': transforms.Compose(__train + __transformations),
                       'test': transforms.Compose(__test + __transformations)
                       }



def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")    """
    model = MyVGG(make_layers(cfg['A']), **kwargs)
    return model

