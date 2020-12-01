import os
from carbongym import gymapi

CARBONGYM_PATH = os.path.dirname(gymapi.__file__)
CARBONGYM_ASSETS_PATH = os.path.join(CARBONGYM_PATH, '..', '..', 'assets')