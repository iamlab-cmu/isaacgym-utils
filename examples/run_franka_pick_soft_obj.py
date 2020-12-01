import argparse

import numpy as np
import matplotlib.pyplot as plt
from autolab_core import YamlConfig, RigidTransform

from carbongym import gymapi
from carbongym_utils.scene import GymScene
from carbongym_utils.assets import GymFranka, GymBoxAsset, GymTetGridAsset
from carbongym_utils.math_utils import RigidTransform_to_transform
from carbongym_utils.policy import GraspPointPolicy
from carbongym_utils.draw import draw_transforms


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

    table_pose = RigidTransform_to_transform(RigidTransform(
        translation=[cfg['table']['dims']['width']/3, cfg['table']['dims']['height']/2, 0]
    ))
    franka_pose = RigidTransform_to_transform(RigidTransform(
        translation=[0, cfg['table']['dims']['height'] + 0.01, 0],
        rotation=RigidTransform.quaternion_from_axis_angle([-np.pi/2, 0, 0])
    ))
    softgrid_pose = gymapi.Transform(p=gymapi.Vec3(0.3, 0.52, -0.025))    

    def custom_draws(scene):
        for i, env_ptr in enumerate(scene.env_ptrs):
            ee_transform = franka.get_ee_transform(env_ptr, 'franka0')
            desired_ee_transform = franka.get_desired_ee_transform(i, 'franka0')

            draw_transforms(scene.gym, scene.viewer, [env_ptr], [ee_transform, desired_ee_transform])

    scene.add_asset('table0', table, table_pose)
    scene.add_asset('franka0', franka, franka_pose, collision_filter=2) # avoid self-collision
    scene.add_asset('softgrid0', softgrid, softgrid_pose)

    ee_pose = franka.get_ee_transform(scene.env_ptrs[0], 'franka0')
    softgrid_grasp_pose = gymapi.Transform(p=softgrid_pose.p, r=ee_pose.r)
    softgrid_grasp_pose.p.z = 0
    softgrid_grasp_pose.p.x = 0.35

    policy = GraspPointPolicy(franka, 'franka0', softgrid_grasp_pose)
    scene.run(policy=policy, custom_draws=custom_draws)
