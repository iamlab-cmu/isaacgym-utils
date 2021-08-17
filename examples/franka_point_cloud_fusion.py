import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from autolab_core import YamlConfig, RigidTransform, PointCloud

from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.camera import GymCamera
from isaacgym_utils.assets import GymFranka, GymBoxAsset
from isaacgym_utils.math_utils import RigidTransform_to_transform

from visualization.visualizer3d import Visualizer3D as vis3d


def vis_cam_images(image_list):
    for i in range(0, len(image_list)):
        plt.figure()
        plt.imshow(image_list[i].data)
    plt.show()


def subsample(pts, rate):
    n = int(rate * len(pts))
    idxs = np.arange(len(pts))
    np.random.shuffle(idxs)
    return pts[idxs[:n]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/franka_point_cloud_fusion.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    # Make scene
    scene = GymScene(cfg['scene'])
    
    table = GymBoxAsset(scene, **cfg['table']['dims'], 
                        shape_props=cfg['table']['shape_props'], 
                        asset_options=cfg['table']['asset_options']
                        )
    franka = GymFranka(cfg['franka'], scene,actuation_mode='attractors')
    table_transform = gymapi.Transform(p=gymapi.Vec3(cfg['table']['dims']['sx']/3, 0, cfg['table']['dims']['sz']/2))
    franka_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, cfg['table']['dims']['sz'] + 0.01))
    
    # Add cameras
    cam = GymCamera(scene, cam_props=cfg['camera'])
    cam_names = [f'cam{i}' for i in range(3)]
    
    def setup(scene, _):
        scene.add_asset('table0', table, table_transform)
        scene.add_asset('franka0', franka, franka_transform, collision_filter=2) # avoid self-collision

        # front
        scene.add_standalone_camera(cam_names[0], cam, RigidTransform_to_transform(
            RigidTransform(
                translation=[1.38, 0, 1],
                rotation=np.array([
                    [0, 0, -1],
                    [1, 0, 0],
                    [0, -1, 0]
                ]) @ RigidTransform.x_axis_rotation(np.deg2rad(-45))
        )))
        # left
        scene.add_standalone_camera(cam_names[1], cam, RigidTransform_to_transform(
            RigidTransform(
                translation=[0.5, -0.8, 1],
                rotation=np.array([
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]
                ]) @ RigidTransform.x_axis_rotation(np.deg2rad(-45))
        )))
        # right
        scene.add_standalone_camera(cam_names[2], cam, RigidTransform_to_transform(
            RigidTransform(
                translation=[0.5, 0.8, 1],
                rotation=np.array([
                    [-1, 0, 0],
                    [0, 0, -1],
                    [0, -1, 0]
                ]) @ RigidTransform.x_axis_rotation(np.deg2rad(-45))
        )))
    scene.setup_all_envs(setup)

    # Render images
    scene.render_cameras()
    color_list, depth_list = [], []
    env_idx = 0
    for cam_name in cam_names:
        # get images of cameras in first env 
        color, depth, _ = cam.frames(env_idx, cam_name)
        color_list.append(color)
        depth_list.append(depth)

    # Plot color and depth images
    vis_cam_images(color_list)
    vis_cam_images(depth_list) 

    # Get camera intrinsics
    intrs = [cam.get_intrinsics(cam_name) for cam_name in cam_names]

    # Deproject to point clouds
    pcs_cam = []
    for i, depth in enumerate(depth_list):
        pc_raw = intrs[i].deproject(depth)
        points_filtered = pc_raw.data[:, np.logical_not(np.any(pc_raw.data > 5, axis=0))]
        pcs_cam.append(PointCloud(points_filtered, pc_raw.frame))

    # Get camera poses
    camera_poses = [
        cam.get_extrinsics(env_idx, cam_name)
        for cam_name in cam_names
    ]

    # Transform point clouds from camera frame into world frame
    pcs_world = [camera_poses[i] * pc for i, pc in enumerate(pcs_cam)]

    # Visualize origin pose, camera poses, and point clouds in world frame
    vis3d.figure()
    vis3d.pose(RigidTransform())
    for camera_pose in camera_poses:
        vis3d.pose(camera_pose)
    for i, pc in enumerate(pcs_world):
        vis3d.points(
            subsample(pc.data.T, 0.1), 
            color=cm.tab10.colors[i],
            scale=0.005
        )
    vis3d.show()

    import IPython; IPython.embed(); exit(0)
