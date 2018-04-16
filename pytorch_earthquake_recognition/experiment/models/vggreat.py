from torchvision.models.vgg import make_layers, VGG, cfg, model_zoo, model_urls
from mytransforms.transforms import NormalizeGray
import torchvision.transforms as transforms

class MyVGG(VGG):
    _transformations = [transforms.Grayscale(num_output_channels=3),
                         transforms.ToTensor(),
                         NormalizeGray]

    _train = [transforms.Resize((200, 310)),
               transforms.RandomCrop((200, 270))
               ]

    _test = [transforms.Resize((200, 310)),
              transforms.RandomCrop((200, 270))
              ]

    transformations = {'train': transforms.Compose(_train + _transformations),
                       'test': transforms.Compose(_test + _transformations)
                       }



def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")    """
    model = MyVGG(make_layers(cfg['A']), **kwargs)
    return model

