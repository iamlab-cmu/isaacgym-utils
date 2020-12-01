import argparse

import numpy as np
from autolab_core import YamlConfig, RigidTransform

from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymFranka, GymBoxAsset
from isaacgym_utils.camera import GymCamera, CameraZMQPublisher
from isaacgym_utils.math_utils import RigidTransform_to_transform, vec3_to_np
from isaacgym_utils.policy import GraspBlockPolicy
from isaacgym_utils.draw import draw_transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/run_franka_pick_block.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    scene = GymScene(cfg['scene'])

    table = GymBoxAsset(scene.gym, scene.sim, **cfg['table']['dims'], 
                        shape_props=cfg['table']['shape_props'], 
                        asset_options=cfg['table']['asset_options']
                        )
    franka = GymFranka(cfg['franka'], scene.gym, scene.sim, actuation_mode='attractors')
    block = GymBoxAsset(scene.gym, scene.sim, **cfg['block']['dims'], 
                        shape_props=cfg['block']['shape_props'], 
                        rb_props=cfg['block']['rb_props'],
                        asset_options=cfg['block']['asset_options']
                        )

    table_pose = RigidTransform_to_transform(RigidTransform(
        translation=[cfg['table']['dims']['width']/3, cfg['table']['dims']['height']/2, 0]
    ))
    franka_pose = RigidTransform_to_transform(RigidTransform(
        translation=[0, cfg['table']['dims']['height'] + 0.01, 0],
        rotation=RigidTransform.quaternion_from_axis_angle([-np.pi/2, 0, 0])
    ))
    
    scene.add_asset('table0', table, table_pose)
    scene.add_asset('franka0', franka, franka_pose, collision_filter=2) # avoid self-collisions
    scene.add_asset('block0', block, gymapi.Transform()) # we'll sample block poses later

    cam = GymCamera(scene.gym, scene.sim, cam_props=cfg['camera'])
    cam_offset_transform = RigidTransform_to_transform(RigidTransform(
        rotation=RigidTransform.x_axis_rotation(-np.pi/2) @ RigidTransform.z_axis_rotation(-np.pi/2),
        translation=np.array([-0.046490, -0.083270, 0])
    ))
    scene.attach_camera('hand_cam0', cam, 'franka0', 'panda_hand', offset_transform=cam_offset_transform)
    cam_pub = CameraZMQPublisher()

    def custom_draws(scene):
        for env_idx, env_ptr in enumerate(scene.env_ptrs):
            ee_transform = franka.get_ee_transform(env_ptr, 'franka0')
            desired_ee_transform = franka.get_desired_ee_transform(env_idx, 'franka0')

            transforms = [ee_transform, desired_ee_transform]

            if 'hand_cam0' in scene.ch_map[env_idx]:
                ch = scene.ch_map[env_idx]['hand_cam0']
                cam_transform = cam.get_transform(ch, env_idx)
                transforms.append(cam_transform)

            draw_transforms(scene.gym, scene.viewer, [env_ptr], transforms)

    def cb(scene, _, __):
        scene.render_cameras()
        color, depth, seg = cam.frames(scene.ch_map[0]['hand_cam0'], 'hand_cam0')
        cam_pub.pub(color, depth, seg)

    policy = GraspBlockPolicy(franka, 'franka0', block, 'block0')

    while True:
        # sample block poses
        block_poses = [RigidTransform(
            translation=[(np.random.rand()*2 - 1) * 0.1 + 0.5, 
            cfg['table']['dims']['height'] + cfg['block']['dims']['height'] / 2 + 0.1, 
            (np.random.rand()*2 - 1) * 0.2
            ]
        ) for _ in range(scene.n_envs)]

        # set block poses
        for i, env_ptr in enumerate(scene.env_ptrs):
            block.set_rb_rigid_transforms(env_ptr, scene.ah_map[i]['block0'], [block_poses[i]])

        policy.reset()
        scene.run(time_horizon=policy.time_horizon, policy=policy, custom_draws=custom_draws, cb=cb)
