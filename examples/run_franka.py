import argparse

import numpy as np
from autolab_core import YamlConfig, RigidTransform

from carbongym import gymapi
from carbongym_utils.scene import GymScene
from carbongym_utils.assets import GymFranka, GymBoxAsset
from carbongym_utils.math_utils import RigidTransform_to_transform
from carbongym_utils.policy import RandomDeltaJointPolicy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/run_franka.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    scene = GymScene(cfg['scene'])
    
    table = GymBoxAsset(scene.gym, scene.sim, **cfg['table']['dims'], 
                        shape_props=cfg['table']['shape_props'], 
                        asset_options=cfg['table']['asset_options']
                        )
    franka = GymFranka(cfg['franka'], scene.gym, scene.sim)

    table_pose = RigidTransform_to_transform(RigidTransform(
        translation=[cfg['table']['dims']['width']/3, cfg['table']['dims']['height']/2, 0]
    ))
    franka_pose = RigidTransform_to_transform(RigidTransform(
        translation=[0, cfg['table']['dims']['height'] + 0.01, 0],
        rotation=RigidTransform.quaternion_from_axis_angle([-np.pi/2, 0, 0])
    ))
    
    scene.add_asset('table0', table, table_pose)
    scene.add_asset('franka0', franka, franka_pose, collision_filter=2) # avoid self-collision

    policy = RandomDeltaJointPolicy(franka, 'franka0')
    scene.run(policy=policy)
