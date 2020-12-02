import argparse

import numpy as np
import matplotlib.pyplot as plt
from autolab_core import YamlConfig, RigidTransform

from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymFranka, GymBoxAsset, GymTetGridAsset
from isaacgym_utils.math_utils import RigidTransform_to_transform
from isaacgym_utils.policy import GraspPointPolicy
from isaacgym_utils.draw import draw_transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/run_franka_pick_soft_obj.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    scene = GymScene(cfg['scene'])
    
    table = GymBoxAsset(scene.gym, scene.sim, **cfg['table']['dims'], 
                        shape_props=cfg['table']['shape_props'], 
                        asset_options=cfg['table']['asset_options']
                        )
    franka = GymFranka(cfg['franka'], scene.gym, scene.sim, actuation_mode='attractors')
    softgrid = GymTetGridAsset(scene.gym, scene.sim, **cfg['softbody']['dims'], 
                            soft_material_props=cfg['softbody']['soft_material_props'],
                            asset_options=cfg['softbody']['asset_options'],
                            shape_props=cfg['softbody']['shape_props']
                            )

    table_transform = gymapi.Transform(p=gymapi.Vec3(cfg['table']['dims']['sx']/3, 0, cfg['table']['dims']['sz']/2))
    franka_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, cfg['table']['dims']['sz'] + 0.01))
    softgrid_pose = gymapi.Transform(p=gymapi.Vec3(0.3, -0.025, 0.52))    

    def custom_draws(scene):
        for env_idx in scene.env_idxs:
            ee_transform = franka.get_ee_transform(env_idx, 'franka0')
            desired_ee_transform = franka.get_desired_ee_transform(env_idx, 'franka0')

            draw_transforms(scene, [env_idx], [ee_transform, desired_ee_transform])

    def setup(scene, _):
        scene.add_asset('table0', table, table_transform)
        scene.add_asset('franka0', franka, franka_transform, collision_filter=2) # avoid self-collision
        scene.add_asset('softgrid0', softgrid, softgrid_pose)
    scene.setup_all_envs(setup)

    ee_pose = franka.get_ee_transform(0, 'franka0')
    softgrid_grasp_pose = gymapi.Transform(p=softgrid_pose.p, r=ee_pose.r)
    softgrid_grasp_pose.p.y = 0
    softgrid_grasp_pose.p.x = 0.35

    policy = GraspPointPolicy(franka, 'franka0', softgrid_grasp_pose)
    scene.run(policy=policy, custom_draws=custom_draws)
