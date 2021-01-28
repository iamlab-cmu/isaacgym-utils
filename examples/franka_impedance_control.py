import argparse

import numpy as np
import quaternion
from autolab_core import YamlConfig

from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymFranka
from isaacgym_utils.policy import EEImpedanceWaypointPolicy
from isaacgym_utils.draw import draw_transforms
from isaacgym_utils.math_utils import min_jerk, vec3_to_np, quat_to_np, angle_axis_between_quats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/franka_impedance_control.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    scene = GymScene(cfg['scene'])
    franka = GymFranka(cfg['franka'], scene, actuation_mode='torques')
    franka_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, 0.01))
    franka_name = 'franka'

    def setup(scene, _):
        scene.add_asset(franka_name, franka, franka_transform, collision_filter=2) # avoid self-collision
    scene.setup_all_envs(setup)

    def custom_draws(scene):
        for env_idx in scene.env_idxs:
            transforms = [franka_transform, franka.get_ee_transform(env_idx, franka_name), 
                        franka.get_links_transforms(env_idx, franka_name)[3]]
            draw_transforms(scene, [env_idx], transforms, length=0.2)

    init_ee_transform = franka.get_ee_transform(0, franka_name)
    goal_ee_transform = gymapi.Transform(
        p=init_ee_transform.p + gymapi.Vec3(0.2, 0.2, -0.4),
        r=init_ee_transform.r
    )
    policy = EEImpedanceWaypointPolicy(franka_name, init_ee_transform, goal_ee_transform)

    scene.run(policy=policy, custom_draws=custom_draws)
