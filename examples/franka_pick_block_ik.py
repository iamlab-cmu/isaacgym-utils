import argparse

import numpy as np
from autolab_core import YamlConfig, RigidTransform

from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymFranka, GymBoxAsset
from isaacgym_utils.camera import GymCamera
from isaacgym_utils.math_utils import (
    np_to_vec3,
    RigidTransform_to_transform,
    rpy_to_quat,
    transform_to_np_rpy,
    vec3_to_np,
)
from isaacgym_utils.policy import GraspBlockPolicy
from isaacgym_utils.draw import draw_transforms, draw_contacts, draw_camera


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", "-c", type=str, default="cfg/franka_pick_block.yaml")
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    scene = GymScene(cfg["scene"])

    table = GymBoxAsset(
        scene,
        **cfg["table"]["dims"],
        shape_props=cfg["table"]["shape_props"],
        asset_options=cfg["table"]["asset_options"]
    )
    franka = GymFranka(cfg["franka"], scene, actuation_mode="attractors")
    block = GymBoxAsset(
        scene,
        **cfg["block"]["dims"],
        shape_props=cfg["block"]["shape_props"],
        rb_props=cfg["block"]["rb_props"],
        asset_options=cfg["block"]["asset_options"]
    )

    table_transform = gymapi.Transform(
        p=gymapi.Vec3(cfg["table"]["dims"]["sx"] / 3, 0, cfg["table"]["dims"]["sz"] / 2)
    )
    franka_transform = gymapi.Transform(
        p=gymapi.Vec3(0, 0, cfg["table"]["dims"]["sz"] + 0.01)
    )

    table_name, franka_name, block_name = "table0", "franka0", "block0"

    cam = GymCamera(scene, cam_props=cfg["camera"])
    cam_offset_transform = RigidTransform_to_transform(
        RigidTransform(
            rotation=RigidTransform.z_axis_rotation(np.deg2rad(90))
            @ RigidTransform.x_axis_rotation(np.deg2rad(1)),
            translation=np.array([-0.083270, -0.046490, 0]),
        )
    )
    cam_name = "hand_cam0"

    def setup(scene, _):
        scene.add_asset(table_name, table, table_transform)
        scene.add_asset(
            franka_name, franka, franka_transform, collision_filter=2
        )  # avoid self-collisions
        scene.add_asset(
            block_name, block, gymapi.Transform()
        )  # we'll sample block poses later
        scene.attach_camera(
            cam_name,
            cam,
            franka_name,
            "panda_hand",
            offset_transform=cam_offset_transform,
        )

    scene.setup_all_envs(setup)

    def custom_draws(scene):
        for env_idx in scene.env_idxs:
            ee_transform = franka.get_ee_transform(env_idx, franka_name)
            # desired_ee_transform = franka.get_desired_ee_transform(env_idx, franka_name)
            transforms = [ee_transform]
            draw_transforms(scene, [env_idx], transforms)
            cam_transform = cam.get_transform(env_idx, cam_name)
            draw_camera(scene, [env_idx], cam_transform, length=0.04)
        draw_contacts(scene, scene.env_idxs)

    def cb(scene, _, __):
        env_idx = 0
        scene.render_cameras()
        frames = cam.frames(env_idx, cam_name)
        cam_pub.pub(frames["color"], frames["depth"], frames["seg"])

    policy = GraspBlockPolicy(franka, franka_name, block, block_name)

    # this was found by setting ee position to [0.4, 0., 0.8] and same initial ee rot
    # this helps IK converge to a sensible solution
    robot_joints_for_ik = np.array(
        [
            1.1024622e-02,
            -4.0491045e-01,
            -1.0826388e-02,
            -2.5944960e00,
            -5.4867277e-03,
            2.1895969e00,
            7.8967488e-01,
        ]
    )

    init_joint_angles = [
        franka.get_joints(env_idx, franka_name) for env_idx in scene.env_idxs
    ]
    init_ee_tforms = [
        franka.get_ee_transform(env_idx, franka_name) for env_idx in scene.env_idxs
    ]

    while True:
        # Reset Franka
        for env_idx in scene.env_idxs:
            franka.set_joints(env_idx, franka_name, init_joint_angles[env_idx])
            franka.set_joints_targets(env_idx, franka_name, init_joint_angles[env_idx])
            franka.set_ee_transform_target(env_idx, franka_name, init_ee_tforms[env_idx])

        # sample block poses
        block_transforms = [
            gymapi.Transform(
                p=gymapi.Vec3(
                    (np.random.rand() * 2 - 1) * 0.1 + 0.5,
                    (np.random.rand() * 2 - 1) * 0.2,
                    cfg["table"]["dims"]["sz"] + cfg["block"]["dims"]["sz"] / 2 + 0.1,
                ),
                r=rpy_to_quat([0.0, 0.0, np.random.uniform(low=-np.pi, high=np.pi)]),
            )
            for _ in range(scene.n_envs)
        ]
        block_colors = np.random.uniform(low=0.0, high=1.0, size=(scene.n_envs, 3))

        # set block poses and colors
        for env_idx in scene.env_idxs:
            block.set_rb_transforms(env_idx, block_name, [block_transforms[env_idx]])
            block.set_rb_props(env_idx, block_name, {"color": block_colors[env_idx]})

        # Pause for a second
        scene.run(time_horizon=100)

        for env_idx in scene.env_idxs:

            this_block_tform = block_transforms[env_idx]

            ee_transform = franka.get_ee_transform(env_idx, franka_name)
            block_rpy = transform_to_np_rpy(this_block_tform)[3:]
            block_vert_angle = block_rpy[2]
            # This wraps to +-pi/2
            ee_vertical_angle = (block_vert_angle + np.pi / 2.0) % (np.pi) - np.pi / 2.0
            ee_rot = rpy_to_quat([0.0, 0.0, ee_vertical_angle])
            ee_orientation = ee_rot * ee_transform.r

            grasp_transform = gymapi.Transform(
                p=this_block_tform.p,
                r=ee_orientation,
            )

            # Now set robot directly using IK
            set_ee_success = franka.set_ee_transform(
                env_idx,
                franka_name,
                grasp_transform,
                ik_robot_joints_hint=robot_joints_for_ik,
            )
            assert (
                set_ee_success
            ), "Could not set EE transform -- IK solution was not found."

        # Pause for a second
        scene.run(time_horizon=100)
