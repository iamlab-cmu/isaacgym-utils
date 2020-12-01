import os
from isaacgym import gymapi

isaacgym_PATH = os.path.dirname(gymapi.__file__)
isaacgym_ASSETS_PATH = os.path.join(isaacgym_PATH, '..', '..', 'assets')