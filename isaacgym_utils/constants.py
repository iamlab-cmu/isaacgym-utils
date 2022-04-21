import numpy as np
import pkg_resources
from pathlib import Path

import isaacgym
from isaacgym import gymapi

import isaacgym_utils


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


# Default joint angles for Franka IK
# this was found by setting ee position to [0.4, 0., 0.8] and same initial ee rot
# this helps IK converge to a sensible solution for most workspace EE poses
franka_default_joints_for_ik = np.array([
    1.1024622e-02,
    -4.0491045e-01,
    -1.0826388e-02,
    -2.5944960e00,
    -5.4867277e-03,
    2.1895969e00,
    7.8967488e-01,
])
