from abc import ABC, abstractmethod
import numpy as np

from autolab_core import RigidTransform

from isaacgym import gymapi
from .math_utils import RigidTransform_to_transform, transform_to_RigidTransform


class Policy(ABC):

    def __init__(self):
        self._time_horizon = -1

    @abstractmethod
    def __call__(self, scene, env_index, t_step, t_sim):
        pass

    def reset(self):
        pass

    @property
    def time_horizon(self):
        return self._time_horizon


class RandomDeltaJointPolicy(Policy):

    def __init__(self, franka, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._franka = franka
        self._name = name

    def __call__(self, scene, env_index, _, __):
        env_ptr = scene.env_ptrs[env_index]
        ah = scene.ah_map[env_index][self._name]

        delta_joints = (np.random.random(self._franka.n_dofs) * 2 - 1) * ([0.05] * 7 + [0.005] * 2)
        self._franka.apply_delta_joint_targets(env_ptr, ah, delta_joints)


class GraspBlockPolicy(Policy):

    def __init__(self, franka, franka_name, block, block_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name

        self._time_horizon = 1000

        self.reset()

    def reset(self):
        self._pre_grasp_transforms = []
        self._grasp_transforms = []
        self._init_ee_transforms = []

    def __call__(self, scene, env_index, t_step, _):
        env_ptr  = scene.env_ptrs[env_index]
        ah = scene.ah_map[env_index][self._franka_name]

        if t_step == 20:
            block_ah = scene.ah_map[env_index][self._block_name]
            block_transform = self._block.get_rb_transforms(env_ptr, block_ah)[0]
            
            ee_transform = self._franka.get_ee_transform(env_ptr, self._franka_name)
            self._init_ee_transforms.append(ee_transform)

            grasp_transform = gymapi.Transform(p=block_transform.p, r=ee_transform.r)
            pre_grasp_transfrom = gymapi.Transform(p=grasp_transform.p, r=grasp_transform.r)
            pre_grasp_transfrom.p.y += 0.2

            self._grasp_transforms.append(grasp_transform)
            self._pre_grasp_transforms.append(pre_grasp_transfrom)

            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_transforms[env_index])

        if t_step == 200:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._grasp_transforms[env_index])

        if t_step == 300:
            self._franka.close_grippers(env_ptr, ah)
        
        if t_step == 400:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_transforms[env_index])

        if t_step == 500:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._grasp_transforms[env_index])

        if t_step == 600:
            self._franka.open_grippers(env_ptr, ah)

        if t_step == 700:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_transforms[env_index])

        if t_step == 800:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._init_ee_transforms[env_index])


class GraspPointPolicy(Policy):

    def __init__(self, franka, franka_name, grasp_transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._franka = franka
        self._franka_name = franka_name
        self._grasp_transform = grasp_transform

        self._time_horizon = 710

        self.reset()

    def reset(self):
        self._pre_grasp_transforms = []
        self._grasp_transforms = []
        self._init_ee_transforms = []

    def __call__(self, scene, env_index, t_step, _):
        env_ptr  = scene.env_ptrs[env_index]
        ah = scene.ah_map[env_index][self._franka_name]

        t_step = t_step % self._time_horizon

        if t_step == 0:
            self._init_joints = self._franka.get_joints(env_ptr, ah)
            self._init_rbs = self._franka.get_rb_states(env_ptr, ah)

        if t_step == 20:
            ee_transform = self._franka.get_ee_transform(env_ptr, self._franka_name)
            self._init_ee_transforms.append(ee_transform)

            pre_grasp_transfrom = gymapi.Transform(p=self._grasp_transform.p, r=self._grasp_transform.r)
            pre_grasp_transfrom.p.y += 0.2

            self._grasp_transforms.append(self._grasp_transform)
            self._pre_grasp_transforms.append(pre_grasp_transfrom)

            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_transforms[env_index])

        if t_step == 100:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._grasp_transforms[env_index])

        if t_step == 150:
            self._franka.close_grippers(env_ptr, ah)
        
        if t_step == 250:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_transforms[env_index])

        if t_step == 350:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._grasp_transforms[env_index])

        if t_step == 500:
            self._franka.open_grippers(env_ptr, ah)

        if t_step == 550:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._pre_grasp_transforms[env_index])

        if t_step == 600:
            self._franka.set_ee_transform(env_ptr, env_index, self._franka_name, self._init_ee_transforms[env_index])

        if t_step == 700:
            self._franka.set_joints(env_ptr, ah, self._init_joints)
            self._franka.set_rb_states(env_ptr, ah, self._init_rbs)
