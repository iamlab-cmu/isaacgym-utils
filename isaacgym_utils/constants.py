import numpy as np
import pkg_resources
from pathlib import Path
import isaacgym
import isaacgym_utils

from isaacgym import gymapi

isaacgym_PATH = Path(isaacgym.__file__).parent.parent
isaacgym_ASSETS_PATH = isaacgym_PATH.parent / 'assets'

isaacgym_utils_PATH = Path(isaacgym_utils.__file__).parent.parent
isaacgym_utils_ASSETS_PATH = isaacgym_utils_PATH / 'assets'

isaacgym_VERSION = pkg_resources.working_set.find(
    pkg_resources.Requirement('isaacgym')
    ).version

# This is to convert between canonical/real/optical cam frame and gym cam frame
# Real cam frame is (z forward, x right, y down)
# Gym cam frame is (x forward, y left, z up)
# R_gym = R_real * R_real_to_gym_cam
R_real_to_gym_cam = np.array([
    [ 0., -1.,  0.],
    [ 0.,  0., -1.],
    [ 1.,  0.,  0.],
])
quat_real_to_gym_cam = gymapi.Quat(0.5, -0.5, 0.5, 0.5)
quat_gym_to_real_cam = quat_real_to_gym_cam.inverse()
