from time import sleep
from multiprocessing import Queue, Process
from queue import Empty

import numpy as np
from simple_zmq import SimpleZMQPublisher
from autolab_core import RigidTransform
from perception import CameraIntrinsics, ColorImage, DepthImage, SegmentationImage

from isaacgym import gymapi
from .math_utils import vec3_to_np, transform_to_RigidTransform, quat_to_rot, quat_to_rot
from .constants import quat_gym_to_real_cam


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

    @property
    def width(self):
        return self.gym_cam_props.width

    @property
    def height(self):
        return self.gym_cam_props.height

    @property
    def fov(self):
        return np.deg2rad(self.gym_cam_props.horizontal_fov)

    @staticmethod
    def optical_from_gym_transform(transform):
        # re-orients the camera in an "optical" convention
        # given the transform provided by isaac gym
        # +z forward, +x right, +y down
        transform.r = transform.r * quat_gym_to_real_cam
        return transform

    def get_transform(self, env_idx, name):
        env_ptr = self._scene.env_ptrs[env_idx]
        ch = self._scene.ch_map[env_idx][name]
        transform = self.optical_from_gym_transform(self._scene.gym.get_camera_transform(self._scene.sim, env_ptr, ch))
        return transform

    def get_extrinsics(self, env_idx, name):
        transform = self.get_transform(env_idx, name)
        return transform_to_RigidTransform(transform, name, 'world')

    def get_intrinsics(self, name):
        hx, hy = self.width/2, self.height/2
        fx = hx / np.tan(self.fov/2)
        fy = fx
        return CameraIntrinsics(name, fx, fy, hx, hy, height=self.height, width=self.width)

    def frames(self, env_idx, name):
        env_ptr = self._scene.env_ptrs[env_idx]
        ch = self._scene.ch_map[env_idx][name]
        raw_color = self._scene.gym.get_camera_image(self._scene.sim, env_ptr, ch, gymapi.IMAGE_COLOR)
        raw_depth = self._scene.gym.get_camera_image(self._scene.sim, env_ptr, ch, gymapi.IMAGE_DEPTH)
        raw_seg = self._scene.gym.get_camera_image(self._scene.sim, env_ptr, ch, gymapi.IMAGE_SEGMENTATION)

        color = _process_gym_color(raw_color)
        depth = _process_gym_depth(raw_depth)

        return ColorImage(color, frame=name), DepthImage(depth, frame=name), SegmentationImage(raw_seg.astype('uint16'), frame=name)


def _process_gym_color(raw_im):
    return raw_im.flatten().reshape(raw_im.shape[0], raw_im.shape[1]//4, 4)[:, :, :3]


def _process_gym_depth(raw_depth, flip=True):
    return raw_depth * (-1 if flip else 1)


class CameraZMQPublisher:

    def __init__(self, topic='gym_cameras', ip='127.0.0.1', port='5555'):
        self._pub = SimpleZMQPublisherAsync(ip, port, topic)
        self._pub.start()

    def pub(self, color, depth, seg):
        camera_data = {
            'color': color.data,
            'depth': depth.data,
            'seg': seg.data
        }

        self._pub.push(camera_data)


class SimpleZMQPublisherAsync(Process):

    def __init__(self, ip, port, topic):
        super().__init__()
        self._ip = ip
        self._port = port
        self._topic = topic

        self._data_q = Queue(maxsize=1)

    def run(self):
        pub = SimpleZMQPublisher(self._ip, self._port, self._topic)

        data = None
        while True:
            try:
                data = self._data_q.get_nowait()
            except Empty:
                pass

            if data is not None:
                pub.push(data)
            sleep(1e-3)

    def push(self, data):
        try:
            self._data_q.get_nowait()
        except Empty:
            pass
        self._data_q.put(data)