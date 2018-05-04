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

everywhere_folders = ['-31.203699--71.000298', '-33.0229--71.637398', '-35.009899--71.930298', '17.969427--67.04422', '18.018099--66.022209', 
'18.979361--155.667892', '19.480289--154.888565', '20.125248--155.777374', '32.820301--117.056702', '32.891998--116.422302', '33.029999--116.085297', 
'33.144199--116.119301', '33.210098--116.409103', '33.260193--116.322289', '33.260201--116.322304', '33.457699--117.170799', '33.495499--116.583397', 
'33.509701--116.561501', '33.539398--116.592201', '33.5397--116.591698', '33.651501--116.739403', '33.6688--116.672997', '36.956821--97.96302', 
'37.012882--97.477806', '37.044071--97.764748', '37.136131--97.618317', '37.225616--98.06472', '39.539101--119.813797', '46.551701--119.645317', 
'-30.172701--70.799301', '-30.8389--70.689102'] + \
['Puerto', 'OklahomaQuakes', 'new-spectrograms-oklahoma', 'AmatriceQuakes', 'SouthAmerica']

my_folders = ['AmatriceQuakes',  'OklahomaQuakes', 'CaliforniaQuakes', 'PuertoQuakes', 
'SouthAmericaQuakes']

use = 'SouthAmerica'
everywhere_folders.remove(use)
ignore_everywhere = [use]

options = dict(
    my_data={
        'train': {
            'path': 'everywhere-97',
            'divide_test': 0,
            'ignore': 'spectrograms/AmatriceQuakes',
        },
        'test': {
            'path': 'everywhere-97',
            'divide_test': 1.0,
            'ignore': everywhere_folders
        },
        'image': {
          'height': int(217 * 1),
          'width': int(296 * 1), # * 1.5 stretch factor to make pixels light up more before the resize
          'crop':  (1, .8),  # height, width   (.6, .8)  (0, 0, .4, 0)
          'padding': (0, 0, 0, 0)    # left, right, top, bottom
        },
    },
    everywhere={
        'train': {
            'path': 'everywhere-97',
            'divide_test': 0,
            'ignore': ignore_everywhere,
        },
        'test': {
            'path': 'everywhere-97',
            'divide_test': 1.0,
            'ignore': everywhere_folders
        },
        'image': {
          'height': int(258 * 1),
          'width': int(293 * 1), # * 1.5 stretch factor to make pixels light up more before the resize
          'crop':  (1, .8),  # height, width   (.6, .8)  (0, 0, .4, 0)
          'padding': (0, 0, 0, 0)    # left, right, top, bottom
        },
    },
    test={
        'train': {
            'path': f'everywhere-97/{use}',
            'divide_test': .2,
        },
        'test': {
            'path': f'everywhere-97/{use}',
            'divide_test': .2,
        },
        'image': {
          'height': int(258 * 1),
          'width': int(293 * 1), # * 1.5 stretch factor to make pixels light up more before the resize
          'crop':  (1, .8),  # height, width   (.6, .8)  (0, 0, .4, 0)
          'padding': (0, 0, 0, 0)    # left, right, top, bottom
        },
        'loader': 'direct'
    })  

# Path to configuration file
default_config_path = os.path.join(os.getcwd(), 'validator/config.json')
configuration = os.environ.get('CONFIGURATION', default_config_path)
options['environment'] = json.loads(open(configuration).read())
