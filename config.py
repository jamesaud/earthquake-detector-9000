import os
from enum import Enum
import json
from copy import deepcopy


# These values were gathered from the Benz Dataset in scratch.py

RGB_MEAN = [0.0009950225259743484, 0.000795141388388241, 0.0018111652018977147]
RGB_STD = [0.0001881388618665583, 0.0006368028766968671, 0.00028853512862149407]

GRAY_MEAN = [0.0009710071717991548, 0.0009710071717991548, 0.0009710071717991548]
GRAY_STD = [0.00037422115896262377, 0.00037422115896262377, 0.00037422115896262377]

BORDER_COLOR_GRAY = 38  # 30
BORDER_COLOR_RGB = (68, 2, 85)  # R, G, B
# Important to get right, for adding noise to spectrograms

VISUALIZE_PATH = os.path.join(os.path.join(os.getcwd(), 'visualize/'))


# Initialize Cross Validation Generalized Detection
path = 'all-spectrograms-symlinks/97'
everywhere_path = os.path.join(os.getcwd(), 'data', path)
everywhere_folders = os.listdir(everywhere_path) 
test = 'benz'                             # Initial folder for LOO cross validation
everywhere_folders.remove(test)

options = dict(
    everywhere={
        'train': {
            'path': path,
            'divide_test': 0,
            'ignore': [test],           # [test]
        },
        'test': {
            'path': path,
            'divide_test': .1,
            'ignore': everywhere_folders,           # everywhere_folders
        },
        'image': {
          'height': int(258 * 1),
          'width': int(293 * 1),
          'crop':  (1, 1),  # height, width
          'padding': (0, 0, 0, 0)    # left, right, top, bottom
        },
        'weigh_classes': [1, 3],
        'loader': 'custom'
    },
    benz_train_set={
        'train': {
            'path': f'Benz/spectrograms/train_set_benz',
            'divide_test': .999,
        },
        'test': {
            'path': f'Benz/spectrograms/train_set_benz',
            'divide_test': .999,
        },
        'image': {
          'height': int(258 * 1),
          'width': int(293 * 1), # * 1.5 stretch factor to make pixels light up more before the resize
          'crop':  (1, 1),  # height, width   (.6, .8)  (0, 0, .4, 0)
          'padding_train': (0, 0, 0, 0),    # left, right, top, bottom
          'padding_test': (0, 0, 0, 0)  # left, right, top, bottom

        },
        'loader': 'direct',
        'weigh_classes': [1, 3]
    },
    benz_test_set={
        'train': {
            'path': f'Benz/spectrograms/test_set_benz_2',
            'divide_test': .999,
        },
        'test': {
            'path': f'Benz/spectrograms/test_set_benz_2',
            'divide_test': .999,
        },
        'image': {
          'height': int(258 * 1),
          'width': int(293 * 1),
          'crop':  (1, 1),  # height, width   (.6, .8)  (0, 0, .4, 0)
          'padding_train': (0, 0, 0, 0),    # left, right, top, bottom
          'padding_test': (0, 0, 0, 0)    # left, right, top, bottom
        },
        'loader': 'named',
        'weigh_classes': [1, 1]
    },
    benz_experiment_set={
         'train': {
            'path': f'Benz/spectrograms/train_set_benz',
            'divide_test': 0,
        },
        'test': {
            'path': f'Benz/spectrograms/test_set_benz_2',
            'divide_test': .999,
        },
        'image': {
          'height': int(258 * 1),
          'width': int(293 * 1),
          'crop':  (1, 1),  # height, width   (.6, .8)  (0, 0, .4, 0)
          'padding_train': (0, 0, 0, 0),    # left, right, top, bottom
          'padding_test': (0, 0, 0, 0)   # Centered
        },
        'loader': 'direct',
        'weigh_classes': [1, 3]
    },
    continuous_unlabeled_set={
        'train': {
            'path': f'Benz/continuous_data_gs29',
            'divide_test': 1,
        },
        'test': {
            'path': f'Benz/continuous_data_gs29',
            'divide_test': 1,
        },
        'image': {
          'height': int(258 * 1),
          'width': int(293 * 1), # * 1.5 stretch factor to make pixels light up more before the resize
          'crop':  (1, 1),  # height, width   (.6, .8)  (0, 0, .4, 0)
          'padding': (0, 0, 0, 0)    # left, right, top, bottom
        },
        'loader': 'named_timestamp',
    })

# Path to configuration file
default_config_path = os.path.join(os.getcwd(), 'validator/config_crossvalidation.json')
configuration = os.environ.get('CONFIGURATION', default_config_path)
options['environment'] = json.loads(open(configuration).read())

