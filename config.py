import os
from enum import Enum
import json
from copy import deepcopy


# These values were gathered from the Benz Dataset in scratch.py

RGB_MEAN = [0.0009950225259743484, 0.000795141388388241, 0.0018111652018977147]
RGB_STD = [0.0001881388618665583, 0.0006368028766968671, 0.00028853512862149407]

GRAY_MEAN = [0.0008378496941398172, 0.0008378496941398172, 0.0008378496941398172]
GRAY_STD = [0.00033416534755744185, 0.00033416534755744185, 0.00033416534755744185]

BORDER_COLOR_GRAY = 38  # 30
BORDER_COLOR_RGB = (68, 2, 85)  # R, G, B
# Important to get right, for adding noise to spectrograms

VISUALIZE_PATH = os.path.join(os.path.join(os.getcwd(), 'visualize/'))

loader_args = dict(
                   num_workers=8,
                   pin_memory=True,
                   )

# For training on all data
path = 'all-spectrograms-symlinks/99'
everywhere_path = os.path.join(os.getcwd(), 'data', path)
everywhere_folders = os.listdir(everywhere_path) 
test = 'benz'                             
everywhere_folders.remove(test)


options = dict(
    meta_learning={
        'train': {
            'path': everywhere_path,
            'divide_test': .2,
            'ignore': [test],           # [test]
        },
        'test': {
            'path': os.path.join(everywhere_path, test),
            'divide_test': .2,
        },
        'image': {
          'height': int(258 * 1),
          'width': int(293 * 1),
          'crop':  (1, 1),  # height, width
          'padding_train': (0, 0, 0, 0),    # left, right, top, bottom
          'padding_test': (0, 0, 0, 0)  # left, right, top, bottom
        },
        'weigh_classes': [1, 1],
        'loader': 'direct'
    },
    everywhere={
        'train': {
            'path': everywhere_path,
            'divide_test': .05,
            'ignore': [test],           # [test]
        },
        'test': {
            'path': everywhere_path,
            'divide_test': .05,
            'ignore': [test],           # everywhere_folders
        },
        'image': {
          'height': int(258 * 1),
          'width': int(293 * 1),
          'crop':  (1, 1),  # height, width
          'padding_train': (0, 0, 0, 0),    # left, right, top, bottom
          'padding_test': (0, 0, 0, 0)  # left, right, top, bottom

        },
        'weigh_classes': [1, 3],
        'loader': 'custom'
    },
    benz_train_set={
        'train': {
            'path': f'Benz/spectrograms/train_set_benz',
            'divide_test': .2,
        },
        'test': {
            'path': f'Benz/spectrograms/train_set_benz',
            'divide_test': .2,
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
    benz_monthly={
        # ['03-14', '04-14', '05-14', '06-14', '07-14', '08-14']
        'train': {
            'path': 'Benz/spectrograms/benz_monthly',
            'divide_test': .2,
            'ignore': ['08-14'],       
        },
        'test': {
            'path': 'Benz/spectrograms/benz_monthly',
            'divide_test': .2,
            'ignore': ['08-14'],       
        },
        'image': {
          'height': int(258 * 1),
          'width': int(293 * 1),
          'crop':  (1, 1),  # height, width
          'padding_train': (0, 0, 0, 0),    # left, right, top, bottom
          'padding_test': (0, 0, 0, 0)  # left, right, top, bottom

        },
        'weigh_classes': [1, 1],
        'loader': 'custom'
    },
    benz_modified={
        'train': {
            'path': 'Benz/spectrograms/modified_benz_without_08_2014',
            'divide_test': 0,
        },
        'test': {
            'path': 'Benz/spectrograms/modified_benz_08_2014',
            'divide_test': .5,
        },
        'image': {
          'height': int(258 * 1),
          'width': int(293 * 1),
          'crop':  (.6, .95),  # height, width
          'padding_train': (0, 0, .3, 0),    # left, right, top, bottom
          'padding_test': (0.025, 0.025, .4, 0)  # left, right, top, bottom

        },
        'weigh_classes': [1, 1],
        'loader': 'direct'
    },
    continuous_unlabeled_set={
        'train': {
            'path': f'',
            'divide_test': 0,
        },
        'test': {
            'path': f'Benz/continuous_data_gs29_8_2014',   # 1 week of data
            'divide_test': 1,
        },
        'image': {
          'height': int(258 * 1),
          'width': int(293 * 1), 
          'crop':  (1, 1),  # height, width   (.6, .8)  (0, 0, .4, 0)
          'padding_train': (0, 0, 0, 0),    # left, right, top, bottom
          'padding_test': (0, 0, 0, 0)   # # left, right, top, bottom
        },
        'loader': 'named_timestamp',
    })

# Path to configuration file
default_config_path = os.path.join(os.getcwd(), 'validator/config_crossvalidation.json')
configuration = os.environ.get('CONFIGURATION', default_config_path)
options['environment'] = json.loads(open(configuration).read())

samples = [6, 10, 20, 30, 40,  50, 60, 70, 80, 90, 100,  200,  500,  1000,   2000,  4000,  8000,  16000,  32000, 64000, 128000]
epochs =  [30,30, 30, 30, 30,  30, 30, 30, 30, 30,  30,   30,   30,    30,     30,    30,    30,     30,     30,     20,    10]

hyperparam_sample_sizes = tuple(zip(samples, epochs))