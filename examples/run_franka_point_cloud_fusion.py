import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from autolab_core import YamlConfig, RigidTransform

from carbongym import gymapi
from carbongym_utils.scene import GymScene
from carbongym_utils.camera import GymCamera
from carbongym_utils.assets import GymFranka, GymBoxAsset
from carbongym_utils.math_utils import RigidTransform_to_transform

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
    parser.add_argument('--cfg', '-c', type=str, default='cfg/run_franka_point_cloud_fusion.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    # Make scene
    scene = GymScene(cfg['scene'])
    
    table = GymBoxAsset(scene.gym, scene.sim, **cfg['table']['dims'], 
                        shape_props=cfg['table']['shape_props'], 
                        asset_options=cfg['table']['asset_options']
                        )
    franka = GymFranka(cfg['franka'], scene.gym, scene.sim,actuation_mode='attractors')

    table_pose = RigidTransform_to_transform(RigidTransform(
        translation=[cfg['table']['dims']['width']/3, cfg['table']['dims']['height']/2, 0]
    ))
    franka_pose = RigidTransform_to_transform(RigidTransform(
        translation=[0, cfg['table']['dims']['height'] + 0.01, 0],
        rotation=RigidTransform.quaternion_from_axis_angle([-np.pi/2, 0, 0])
    ))

    scene.add_asset('table0', table, table_pose)
    scene.add_asset('franka0', franka, franka_pose, collision_filter=2) # avoid self-collision

    # Add cameras
    cam = GymCamera(scene.gym, scene.sim, cam_props=cfg['camera'])
    scene.add_standalone_camera('cam0', cam, gymapi.Vec3(1.38, 1.0, 0), gymapi.Vec3(1.0, .5, 0)) # front
    scene.add_standalone_camera('cam1', cam, gymapi.Vec3(0.5, 1.0, .8), gymapi.Vec3(.6, 0, 0)) # left
    scene.add_standalone_camera('cam2', cam, gymapi.Vec3(0.5, 1.0, -0.8), gymapi.Vec3(.6, 0, 0)) # right
    cam_names = ['cam{}'.format(i) for i in range(3)]

    # Render images
    scene.render_cameras()
    color_list, depth_list = [], []
    for cam_name in cam_names:
        # get images of cameras in first env 
        color, depth, _ = cam.frames(scene.ch_map[0][cam_name], cam_name)
        color_list.append(color)
        depth_list.append(depth)

    # Plot color and depth images
    vis_cam_images(color_list)
    vis_cam_images(depth_list) 

    # Get camera intrinsics
    intrs = [cam.get_intrinsics(cam_name) for cam_name in cam_names]

    # Deproject to point clouds
    pcs_cam = [intrs[i].deproject(depth) for i, depth in enumerate(depth_list)]

    # Get camera poses
    camera_poses = [
        cam.get_extrinsics(scene.ch_map[0][cam_name], cam_name, scene.env_ptrs[0])
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
