import torch

class dotdict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)


    """dot.notation access to dictionary attributes"""
    def __getattr__(self, attr):
        return self.get(attr)

    __delattr__ = dict.__delitem__

    def __setitem__(self, key, value):
        if type(value) is dict:
            value = dotdict(value)
        return super().__setitem__(key, value)

    def get(self, key):
        value = super().get(key)
        if type(value) is dict:
            self.__setitem__(key, dotdict(value))
            value = self.__getitem__(key)
        return value

class Evaluator:

    def __init__(self, class_accuracy_dict=None):
        self.class_info = dotdict()  # Javasript style item access

        if class_accuracy_dict:
            for key, value in class_accuracy_dict.items():
                self.class_info[key] = value

    def update_accuracy(self, class_name: str, amount_correct: int, amount_total: int):
        self.class_info[class_name] = {
            'amount_correct': amount_correct,
            'amount_total': amount_total
        }

    def class_details(self, class_name):
        return self.class_info[class_name]

    def percent_correct(self, class_name):
        Class = self.class_info[class_name]
        return Class.amount_correct / Class.amount_total

    def total_percent_correct(self):

        amount_correct = 0
        amount_total = 0

        for class_name, info in self.class_info.items():
            amount_correct += info.amount_correct
            amount_total += info.amount_total

        return amount_correct / amount_total

    def __str__(self):
        return 'Evaluator Object: ' + str(self.class_info)


def make_weights_for_classes(images, nclasses, weigh_classes=None):
    count = [0] * nclasses   # how many of each class there are
    items = [images.__getitem__(index, apply_transforms=False) for index in range(len(images))]
    
    weigh_classes = dict(enumerate(weigh_classes)) if weigh_classes else dict()
    for item in items:
        label = item[1]
        count[label] += 1

    weight_per_class = [0.] * nclasses
    N = sum(count)   # Total number of samples

    for i in range(nclasses):
        weight_per_class[i] = N/count[i]                   # Weight is, if 500 samples, 100/500 local and 400/500 quake

    weight = [0] * len(images)
    for idx, val in enumerate(items):
        label = val[1]
        weight[idx] = weight_per_class[label] * weigh_classes.get(label, 1)
    return weight


# Weighted sampler
def make_weighted_sampler(dataset, num_classes, weigh_classes=None) -> torch.utils.data.sampler.WeightedRandomSampler:
    weights = make_weights_for_classes(dataset, num_classes, weigh_classes)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    return sampler



class IntegrityError(Exception):
    pass

def verify_dataset_integrity(*args):
    paths = set()

    for dataset in args:
        for trio in dataset.file_paths:
            for path in trio:
                if path in paths:
                    raise IntegrityError(f'Duplicate file found in datasets "{path}"')
                paths.add(path)

    return True


def lmap(*args, **kwargs):
    return list(map(*args, **kwargs))