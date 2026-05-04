import os

PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MAP_NAME = 'gridnet3x3'
DATA_DIR = 'data'
SUMO_CONF_FILE = os.path.join('maps', MAP_NAME, f'{MAP_NAME}.sumocfg')

SUMO_CONFIG_MANUAL = [
    'sumo-gui',
    '-c', os.path.join(PROJ_DIR, SUMO_CONF_FILE),
    '--step-length', '1',
    '--delay', '40',
    '-S', '-Q'
]
SUMO_CONFIG_AUTO = [
    'sumo',
    '-c', os.path.join(PROJ_DIR, SUMO_CONF_FILE),
    '--step-length', '1',
    '--duration-log.disable',
    '--no-step-log',
    '--no-warnings'
]

CPU_THREADS = 10