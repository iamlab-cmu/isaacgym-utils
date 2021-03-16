import argparse

import numpy as np
from autolab_core import YamlConfig, RigidTransform

from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymFranka, GymBoxAsset
from isaacgym_utils.camera import GymCamera, CameraZMQPublisher
from isaacgym_utils.math_utils import RigidTransform_to_transform
from isaacgym_utils.policy import GraspBlockPolicy
from isaacgym_utils.draw import draw_transforms, draw_contacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/franka_pick_block.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    scene = GymScene(cfg['scene'])

    table = GymBoxAsset(scene, **cfg['table']['dims'], 
                        shape_props=cfg['table']['shape_props'], 
                        asset_options=cfg['table']['asset_options']
                        )
    franka = GymFranka(cfg['franka'], scene, actuation_mode='torques')
    block = GymBoxAsset(scene, **cfg['block']['dims'], 
                        shape_props=cfg['block']['shape_props'], 
                        rb_props=cfg['block']['rb_props'],
                        asset_options=cfg['block']['asset_options']
                        )

    table_transform = gymapi.Transform(p=gymapi.Vec3(cfg['table']['dims']['sx']/3, 0, cfg['table']['dims']['sz']/2))
    franka_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, cfg['table']['dims']['sz'] + 0.01))
    
    table_name, franka_name, block_name = 'table0', 'franka0', 'block0'

    cam = GymCamera(scene, cam_props=cfg['camera'])
    cam_offset_transform = RigidTransform_to_transform(RigidTransform(
        rotation=RigidTransform.z_axis_rotation(np.deg2rad(90)) @ RigidTransform.x_axis_rotation(np.deg2rad(1)),
        translation=np.array([-0.083270, -0.046490, 0])
    ))
    cam_name = 'hand_cam0'
    cam_pub = CameraZMQPublisher()

    def setup(scene, _):
        scene.add_asset(table_name, table, table_transform)
        scene.add_asset(franka_name, franka, franka_transform, collision_filter=2) # avoid self-collisions
        scene.add_asset(block_name, block, gymapi.Transform()) # we'll sample block poses later
        scene.attach_camera(cam_name, cam, franka_name, 'panda_hand', offset_transform=cam_offset_transform)
    scene.setup_all_envs(setup)    

    def custom_draws(scene):
        for env_idx in scene.env_idxs:
            ee_transform = franka.get_ee_transform(env_idx, franka_name)
            # desired_ee_transform = franka.get_desired_ee_transform(env_idx, franka_name)
            cam_transform = cam.get_transform(env_idx, cam_name)

            transforms = [ee_transform, cam_transform]
            draw_transforms(scene, [env_idx], transforms)
        draw_contacts(scene, scene.env_idxs)

    def cb(scene, _, __):
        env_idx = 0
        scene.render_cameras()
        color, depth, seg = cam.frames(env_idx, cam_name)
        cam_pub.pub(color, depth, seg)

    policy = GraspBlockPolicy(franka, franka_name, block, block_name)

    while True:
        # sample block poses
        block_transforms = [gymapi.Transform(p=gymapi.Vec3(
            (np.random.rand()*2 - 1) * 0.1 + 0.5, 
            (np.random.rand()*2 - 1) * 0.2,
            cfg['table']['dims']['sz'] + cfg['block']['dims']['sz'] / 2 + 0.1
        )) for _ in range(scene.n_envs)]

        # set block poses
        for env_idx in scene.env_idxs:
            block.set_rb_transforms(env_idx, block_name, [block_transforms[env_idx]])

        policy.reset()
        scene.run(time_horizon=policy.time_horizon, policy=policy, custom_draws=custom_draws, cb=cb)
