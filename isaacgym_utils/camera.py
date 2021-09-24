from copy import deepcopy
import quaternion

import numpy as np
from autolab_core import CameraIntrinsics, ColorImage, DepthImage, SegmentationImage, NormalCloudImage

from isaacgym import gymapi
from .math_utils import transform_to_RigidTransform, vec3_to_np, np_quat_to_quat
from .constants import quat_gym_to_real_cam, quat_real_to_gym_cam


class GymCamera:

    def __init__(self, scene, cam_props={}):
        self._scene = scene

        self.gym_cam_props = gymapi.CameraProperties()
        for key, value in cam_props.items():
            setattr(self.gym_cam_props, key, value)

        self._x_axis_rot = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ])

        self._intr_map = {}

    @property
    def width(self):
        return self.gym_cam_props.width

    @property
    def height(self):
        return self.gym_cam_props.height

    @property
    def fov(self):
        return np.deg2rad(self.gym_cam_props.horizontal_fov)

    def get_transform(self, env_idx, name):
        env_ptr = self._scene.env_ptrs[env_idx]
        ch = self._scene.ch_map[env_idx][name]
        # re-orients the camera in an "optical" convention
        # given the transform provided by isaac gym
        # +z forward, +x right, +y down
        transform = self._scene.gym.get_camera_transform(self._scene.sim, env_ptr, ch)
        transform.r = transform.r * quat_gym_to_real_cam
        return transform

    def set_look_at(self, env_idx, name, look_from_pos, look_at_pos):
        z_axis = vec3_to_np(look_at_pos - look_from_pos)
        z_axis = z_axis / np.linalg.norm(z_axis)

        y_axis = np.array([0, 0, -1])
        y_axis = y_axis - y_axis @ z_axis * z_axis
        y_axis = y_axis / np.linalg.norm(y_axis)

        x_axis = np.cross(y_axis, z_axis)

        R = np.c_[x_axis, y_axis, z_axis]
        q = np_quat_to_quat( quaternion.from_rotation_matrix(R))

        cam_transform = gymapi.Transform(look_from_pos, q)
        self.set_transform(env_idx, name, cam_transform)

    def set_transform(self, env_idx, name, transform):
        env_ptr = self._scene.env_ptrs[env_idx]
        ch = self._scene.ch_map[env_idx][name]

        tform_gym = deepcopy(transform)
        tform_gym.r = tform_gym.r * quat_real_to_gym_cam

        self._scene.gym.set_camera_transform(ch, env_ptr, tform_gym)

    def get_extrinsics(self, env_idx, name):
        transform = self.get_transform(env_idx, name)
        return transform_to_RigidTransform(transform, name, 'world')

    def get_intrinsics(self, name):
        if name not in self._intr_map:
            hx, hy = self.width/2, self.height/2
            fx = hx / np.tan(self.fov/2)
            fy = fx
            self._intr_map[name] = CameraIntrinsics(name, fx, fy, hx, hy, height=self.height, width=self.width)
        return self._intr_map[name]

    def frames(self, env_idx, name, get_color=True, get_depth=True, get_seg=True, get_normal=True):
        assert get_color or get_depth or get_seg or get_normal
        
        env_ptr = self._scene.env_ptrs[env_idx]
        ch = self._scene.ch_map[env_idx][name]

        frames = {}

        if get_color:
            raw_color = self._scene.gym.get_camera_image(self._scene.sim, env_ptr, ch, gymapi.IMAGE_COLOR)
            color = _process_gym_color(raw_color)
            frames['color'] = ColorImage(color, frame=name)
        if get_depth:
            raw_depth = self._scene.gym.get_camera_image(self._scene.sim, env_ptr, ch, gymapi.IMAGE_DEPTH)
            depth = _process_gym_depth(raw_depth)
            frames['depth'] = DepthImage(depth, frame=name)
        if get_seg:
            raw_seg = self._scene.gym.get_camera_image(self._scene.sim, env_ptr, ch, gymapi.IMAGE_SEGMENTATION)
            frames['seg'] = SegmentationImage(raw_seg.astype('uint16'), frame=name)
        if get_normal:
            if get_depth:
                depth_im = frames['depth']
            else:
                raw_depth = self._scene.gym.get_camera_image(self._scene.sim, env_ptr, ch, gymapi.IMAGE_DEPTH)
                depth = _process_gym_depth(raw_depth)
                depth_im = DepthImage(depth, frame=name)
            
            normal = _make_normal_map(depth_im, self.get_intrinsics(name))
            frames['normal'] = NormalCloudImage(normal, frame=name)
        
        return frames
            

def _process_gym_color(raw_im):
    return raw_im.flatten().reshape(raw_im.shape[0], raw_im.shape[1]//4, 4)[:, :, :3]


def _process_gym_depth(raw_depth, flip=True):
    return raw_depth * (-1 if flip else 1)


def _make_normal_map(depth, intr, inf_depth=100):
    # from https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf
    depth_data = depth.data.copy()
    inf_px_mask = np.isinf(depth_data)
    depth_data[inf_px_mask] = inf_depth
    depth = DepthImage(depth_data, frame=depth.frame)

    pts = intr.deproject_to_image(depth).data
    
    A = pts[1:, :-1] - pts[:-1, :-1]
    B = pts[:-1, 1:] - pts[:-1, :-1]
    C = np.cross(A.reshape(-1, 3), B.reshape(-1, 3))
    D = C / np.linalg.norm(C, axis=1).reshape(-1, 1)
    
    normal = np.zeros((depth.shape[0], depth.shape[1], 3))
    normal[:-1, :-1] = D.reshape(A.shape)
    normal[-1, :] = normal[-2, :]
    normal[:, -1] = normal[:, -2]
    normal[inf_px_mask] = 0

    return normal
