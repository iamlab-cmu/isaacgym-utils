from pathlib import Path
import isaacgym
import isaacgym_utils

isaacgym_PATH = Path(isaacgym.__file__).parent.parent
isaacgym_ASSETS_PATH = isaacgym_PATH.parent / 'assets'

isaacgym_utils_PATH = Path(isaacgym_utils.__file__).parent.parent
isaacgym_utils_ASSETS_PATH = isaacgym_utils_PATH / 'assets'