import os
from enum import Enum
import json
from copy import deepcopy


# These values were gathered from the Benz Dataset in scratch.py

RGB_MEAN = [0.0009950225259743484, 0.000795141388388241, 0.0018111652018977147]
RGB_STD = [0.0001881388618665583, 0.0006368028766968671, 0.00028853512862149407]

GRAY_MEAN = [0.0009710071717991548, 0.0009710071717991548, 0.0009710071717991548]
GRAY_STD = [0.00037422115896262377, 0.00037422115896262377, 0.00037422115896262377]

BORDER_COLOR_GRAY = 30   #
# Important to get right, for adding noise to spectrograms

VISUALIZE_PATH = os.path.join(os.path.join(os.getcwd(), 'visualize/'))



path = 'everywhere-97'
everywhere_path = os.path.join(os.getcwd(), 'data', path)
everywhere_folders = os.listdir(everywhere_path)
test = 'AmatriceQuakes'
everywhere_folders.remove(test)

options = dict(
    everywhere={
        'train': {
            'path': path,
            'divide_test': 0,
            'ignore': [test],
        },
        'test': {
            'path': path,
            'divide_test': 1.0,
            'ignore': everywhere_folders
        },
        'image': {
          'height': int(258 * 1),
          'width': int(293 * 1), # * 1.5 stretch factor to make pixels light up more before the resize
          'crop':  (1, .8),  # height, width   (.6, .8)  (0, 0, .4, 0)
          'padding': (.1, 0, 0, 0)    # left, right, top, bottom
        },
        'weigh_classes': [4, 1]
    },
    benz_train_set_old={
        'train': {
            'path': f'Benz/trainset-old/spectrograms-train-gs29-original',
            'divide_test': .2,
        },
        'test': {
            'path': f'Benz/trainset-old/spectrograms-train-gs29-original',
            'divide_test': .2,
        },
        'image': {
          'height': int(258 * 1),
          'width': int(293 * 1), # * 1.5 stretch factor to make pixels light up more before the resize
          'crop':  (1, .8),  # height, width   (.6, .8)  (0, 0, .4, 0)
          'padding': (0, .1, 0, 0)    # left, right, top, bottom
        },
        'loader': 'direct',
        'weigh_classes': [4, 1]
    },
    benz_train_set={
        'train': {
            'path': f'Benz/spectrograms/train_set_benz',
            'divide_test': .15,
        },
        'test': {
            'path': f'Benz/spectrograms/train_set_benz',
            'divide_test': .15,
        },
        'image': {
          'height': int(258 * 1),
          'width': int(293 * 1), # * 1.5 stretch factor to make pixels light up more before the resize
          'crop':  (1, .7),  # height, width   (.6, .8)  (0, 0, .4, 0)
          'padding': (0, .1, 0, 0)    # left, right, top, bottom
        },
        'loader': 'direct',
        'weigh_classes': [4, 1]
    },
    benz_test_set={
        'train': {
            'path': f'Benz/spectrograms/test_set_benz',
            'divide_test': .15,
        },
        'test': {
            'path': f'Benz/spectrograms/test_set_benz',
            'divide_test': .15,
        },
        'image': {
          'height': int(258 * 1),
          'width': int(293 * 1),
          'crop':  (1, .8),  # height, width   (.6, .8)  (0, 0, .4, 0)
          'padding': (0, 0, 0, 0)    # left, right, top, bottom
        },
        'loader': 'direct',
        'weigh_classes': [4, 1]
    },
    benz_experiment_set={
        'train': {
            'path': f'Benz/spectrograms-train',
            'divide_test': 0,
        },
        'test': {
            'path': f'Benz/spectrograms-testset-benz/GK29-20-seconds',
            'divide_test': 1,
        },
        'image': {
          'height': int(258 * 1),
          'width': int(293 * 1),
          'crop':  (1, .6),  # height, width   (.6, .8)  (0, 0, .4, 0)
          'padding': (.1, 0, 0, 0)    # left, right, top, bottom
        },
        'loader': 'direct',
        'weigh_classes': [2, 1]
    },
    continuous_unlabeled_set={
        'train': {
            'path': f'Benz/spectrograms-all-29',
            'divide_test': 0,
        },
        'test': {
            'path': f'Benz/spectrograms-all-29',
            'divide_test': 1,
        },
        'image': {
          'height': int(258 * 1),
          'width': int(293 * 1), # * 1.5 stretch factor to make pixels light up more before the resize
          'crop':  (.8, 1),  # height, width   (.6, .8)  (0, 0, .4, 0)
          'padding': (0, 0, 0, 0)    # left, right, top, bottom
        },
        'loader': 'direct',
    })

# Path to configuration file
default_config_path = os.path.join(os.getcwd(), 'validator/config.json')
configuration = os.environ.get('CONFIGURATION', default_config_path)
options['environment'] = json.loads(open(configuration).read())
