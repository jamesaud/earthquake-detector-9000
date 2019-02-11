import torch.nn as nn
import torchvision.transforms as transforms
import mytransforms.transforms as mytransforms


class base_model(nn.Module):
    _transformations = []
    _train = []
    _test = []
    transformations = {'train':  transforms.Compose(_train + _transformations),
                       'test': transforms.Compose(_test + _transformations)
                       }

    


