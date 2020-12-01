from time import sleep
from multiprocessing import Queue, Process
from queue import Empty

import numpy as np
from simple_zmq import SimpleZMQPublisher
from autolab_core import RigidTransform
from perception import CameraIntrinsics, ColorImage, DepthImage, SegmentationImage

from carbongym import gymapi
from .math_utils import vec3_to_np


class GymCamera:

    def __init__(self, gym, sim, cam_props={}):
        self._gym = gym
        self._sim = sim

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

    def get_transform(self, ch, env_idx):
        transform = self._gym.get_camera_transform(self._sim, ch)
        # extrinsics transform to env origin
        env = self._gym.get_env(self._sim, env_idx)
        env_tf = self._gym.get_env_origin(env)
        transform.p -= env_tf # apply environment offset, assuming no rotation between envs
        # rotate 180 degrees about x-axis, default transform is flipped for some reason
        transform.r = transform.r * gymapi.Quat(0,1,0,0) 
        return transform

    def get_extrinsics(self, ch, name, env_ptr):
        IV = np.linalg.inv(self._gym.get_camera_view_matrix(self._sim, ch)).T
        
        R = IV[:3, :3] @ self._x_axis_rot
        T = IV[:3, 3] - vec3_to_np(self._gym.get_env_origin(env_ptr))

        return RigidTransform(rotation=R, translation=T, from_frame=name, to_frame='world')

    def get_intrinsics(self, name):
        hx, hy = self.width/2, self.height/2
        fx = hx / np.tan(self.fov/2)
        fy = fx
        return CameraIntrinsics(name, fx, fy, hx, hy, height=self.height, width=self.width)

    def frames(self, ch, name):
        raw_color = self._gym.get_camera_image(self._sim, ch, gymapi.IMAGE_COLOR)
        raw_depth = self._gym.get_camera_image(self._sim, ch, gymapi.IMAGE_DEPTH)
        raw_seg = self._gym.get_camera_image(self._sim, ch, gymapi.IMAGE_SEGMENTATION)

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