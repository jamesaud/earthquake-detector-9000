import os
from enum import Enum
import json
from copy import deepcopy

RGB_MEAN = [0.0009950225259743484, 0.000795141388388241, 0.0018111652018977147]
RGB_STD = [0.0001881388618665583, 0.0006368028766968671, 0.00028853512862149407]

GRAY_MEAN = [0.0009710071717991548, 0.0009710071717991548, 0.0009710071717991548]
GRAY_STD = [0.00037422115896262377, 0.00037422115896262377, 0.00037422115896262377]

BORDER_COLOR_GRAY = 30

DIVIDE_TEST = .2   # % test

VISUALIZE_PATH = os.path.join(os.path.join(os.getcwd(), 'visualize/'))



path = 'everywhere-once'
everywhere_path = os.path.join(os.getcwd(), 'data', path)
everywhere_folders = os.listdir(everywhere_path)
use = '-22.95196--68.178757'
everywhere_folders.remove(use)

options = dict(
    everywhere={
        'train': {
            'path': path,
            'divide_test': 0,
            'ignore': [use],
        },
        'test': {
            'path': path,
            'divide_test': 1.0,
            'ignore': everywhere_folders
        },
        'image': {
          'height': int(258 * 1),
          'width': int(293 * 1), # * 1.5 stretch factor to make pixels light up more before the resize
          'crop':  (1, 1),  # height, width   (.6, .8)  (0, 0, .4, 0)
          'padding': (0, 0, 0, 0)    # left, right, top, bottom
        },
        'weigh_classes': [7, 1]
    },
    single_location={
        'train': {
            'path': f'spectrograms/new-spectrograms-oklahoma',
            'divide_test': .2,
        },
        'test': {
            'path': f'spectrograms/new-spectrograms-oklahoma',
            'divide_test': .2,
        },
        'image': {
          'height': int(258 * 1),
          'width': int(293 * 1), # * 1.5 stretch factor to make pixels light up more before the resize
          'crop':  (.8, 1),  # height, width   (.6, .8)  (0, 0, .4, 0)
          'padding': (0, 0, 0, 0)    # left, right, top, bottom
        },
        'loader': 'direct',
        'weigh_classes': [10, 1]
    })  

# Path to configuration file
default_config_path = os.path.join(os.getcwd(), 'validator/config.json')
configuration = os.environ.get('CONFIGURATION', default_config_path)
options['environment'] = json.loads(open(configuration).read())
