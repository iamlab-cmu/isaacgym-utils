from abc import ABC, abstractmethod
import numpy as np

from isaacgym import gymapi
from .math_utils import min_jerk, slerp_quat, vec3_to_np, np_to_vec3, \
                    project_to_line, compute_task_space_impedance_control


class Policy(ABC):

    def __init__(self):
        self._time_horizon = -1

    @abstractmethod
    def __call__(self, scene, env_idx, t_step, t_sim):
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

    def __call__(self, scene, env_idx, _, __):
        delta_joints = (np.random.random(self._franka.n_dofs) * 2 - 1) * ([0.05] * 7 + [0.005] * 2)
        self._franka.apply_delta_joint_targets(env_idx, self._name, delta_joints)


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
        self._ee_waypoint_policies = []

    def __call__(self, scene, env_idx, t_step, t_sim):
        ee_transform = self._franka.get_ee_transform(env_idx, self._franka_name)

        if t_step == 0:
            self._init_ee_transforms.append(ee_transform)
            self._ee_waypoint_policies.append(
                EEImpedanceWaypointPolicy(self._franka, self._franka_name, ee_transform, ee_transform, T=20)
            )

        if t_step == 20:
            block_transform = self._block.get_rb_transforms(env_idx, self._block_name)[0]
            grasp_transform = gymapi.Transform(p=block_transform.p, r=self._init_ee_transforms[env_idx].r)
            pre_grasp_transfrom = gymapi.Transform(p=grasp_transform.p + gymapi.Vec3(0, 0, 0.2), r=grasp_transform.r)

            self._grasp_transforms.append(grasp_transform)
            self._pre_grasp_transforms.append(pre_grasp_transfrom)

            self._ee_waypoint_policies[env_idx] = \
                EEImpedanceWaypointPolicy(
                    self._franka, self._franka_name, ee_transform, self._pre_grasp_transforms[env_idx], T=180
                )

        if t_step == 200:
            self._ee_waypoint_policies[env_idx] = \
                EEImpedanceWaypointPolicy(
                    self._franka, self._franka_name, self._pre_grasp_transforms[env_idx], self._grasp_transforms[env_idx], T=100
                )

        if t_step == 300:
            self._franka.close_grippers(env_idx, self._franka_name)
        
        if t_step == 400:
            self._ee_waypoint_policies[env_idx] = \
                EEImpedanceWaypointPolicy(
                    self._franka, self._franka_name, self._grasp_transforms[env_idx], self._pre_grasp_transforms[env_idx], T=100
                )

        if t_step == 500:
            self._ee_waypoint_policies[env_idx] = \
                EEImpedanceWaypointPolicy(
                    self._franka, self._franka_name, self._pre_grasp_transforms[env_idx], self._grasp_transforms[env_idx], T=100
                )

        if t_step == 600:
            self._franka.open_grippers(env_idx, self._franka_name)

        if t_step == 700:
            self._ee_waypoint_policies[env_idx] = \
                EEImpedanceWaypointPolicy(
                    self._franka, self._franka_name, self._grasp_transforms[env_idx], self._pre_grasp_transforms[env_idx], T=100
                )

        if t_step == 800:
            self._ee_waypoint_policies[env_idx] = \
                EEImpedanceWaypointPolicy(
                    self._franka, self._franka_name, self._pre_grasp_transforms[env_idx], self._init_ee_transforms[env_idx], T=100
                )

        self._ee_waypoint_policies[env_idx](scene, env_idx, t_step, t_sim)


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

    def __call__(self, scene, env_idx, t_step, _):
        t_step = t_step % self._time_horizon

        if t_step == 0:
            self._init_joints = self._franka.get_joints(env_idx, self._franka_name)
            self._init_rbs = self._franka.get_rb_states(env_idx, self._franka_name)

        if t_step == 20:
            ee_transform = self._franka.get_ee_transform(env_idx, self._franka_name)
            self._init_ee_transforms.append(ee_transform)

            pre_grasp_transfrom = gymapi.Transform(p=self._grasp_transform.p, r=self._grasp_transform.r)
            pre_grasp_transfrom.p.z += 0.2

            self._grasp_transforms.append(self._grasp_transform)
            self._pre_grasp_transforms.append(pre_grasp_transfrom)

            self._franka.set_ee_transform(env_idx, self._franka_name, self._pre_grasp_transforms[env_idx])

        if t_step == 100:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._grasp_transforms[env_idx])

        if t_step == 150:
            self._franka.close_grippers(env_idx, self._franka_name)
        
        if t_step == 250:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._pre_grasp_transforms[env_idx])

        if t_step == 350:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._grasp_transforms[env_idx])

        if t_step == 500:
            self._franka.open_grippers(env_idx, self._franka_name)

        if t_step == 550:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._pre_grasp_transforms[env_idx])

        if t_step == 600:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._init_ee_transforms[env_idx])

        if t_step == 700:
            self._franka.set_joints(env_idx, self._franka_name, self._init_joints)
            self._franka.set_rb_states(env_idx, self._franka_name, self._init_rbs)


class FrankaEEImpedanceController:

    def __init__(self, franka, franka_name):
        self._franka = franka
        self._franka_name = franka_name
        self._elbow_joint = 3

        Kp_0, Kr_0 = 200, 8
        Kp_1, Kr_1 = 200, 5
        self._Ks_0 = np.diag([Kp_0] * 3 + [Kr_0] * 3)
        self._Ds_0 = np.diag([4 * np.sqrt(Kp_0)] * 3 + [2 * np.sqrt(Kr_0)] * 3)
        self._Ks_1 = np.diag([Kp_1] * 3 + [Kr_1] * 3)
        self._Ds_1 = np.diag([4 * np.sqrt(Kp_1)] * 3 + [2 * np.sqrt(Kr_1)] * 3)

    def compute_tau(self, env_idx, target_transform):
        # primary task - ee control
        ee_transform = self._franka.get_ee_transform(env_idx, self._franka_name)

        J = self._franka.get_jacobian(env_idx, self._franka_name)
        q_dot = self._franka.get_joints_velocity(env_idx, self._franka_name)[:7]
        x_vel = J @ q_dot

        tau_0 = compute_task_space_impedance_control(J, ee_transform, target_transform, x_vel, self._Ks_0, self._Ds_0)

        # secondary task - elbow straight
        link_transforms = self._franka.get_links_transforms(env_idx, self._franka_name)
        elbow_transform = link_transforms[self._elbow_joint]

        u0 = vec3_to_np(link_transforms[0].p)[:2]
        u1 = vec3_to_np(link_transforms[-1].p)[:2]
        curr_elbow_xyz = vec3_to_np(elbow_transform.p)
        goal_elbow_xy = project_to_line(curr_elbow_xyz[:2], u0, u1)
        elbow_target_transform = gymapi.Transform(
            p=gymapi.Vec3(goal_elbow_xy[0], goal_elbow_xy[1], curr_elbow_xyz[2] + 0.2),
            r=elbow_transform.r
        )

        J_elb = self._franka.get_jacobian(env_idx, self._franka_name, target_joint=self._elbow_joint)
        x_vel_elb = J_elb @ q_dot

        tau_1 = compute_task_space_impedance_control(J_elb, elbow_transform, elbow_target_transform, x_vel_elb, self._Ks_1, self._Ds_1)
        
        # nullspace projection
        JT_inv = np.linalg.pinv(J.T)
        Null = np.eye(7) - J.T @ (JT_inv)
        tau = tau_0 + Null @ tau_1

        return tau


class EEImpedanceWaypointPolicy(Policy):

    def __init__(self, franka, franka_name, init_ee_transform, goal_ee_transform, T=300):
        self._franka = franka
        self._franka_name = franka_name

        self._T = T
        self._ee_impedance_ctrlr = FrankaEEImpedanceController(franka, franka_name)

        init_ee_pos = vec3_to_np(init_ee_transform.p)
        goal_ee_pos = vec3_to_np(goal_ee_transform.p)
        self._traj = [
            gymapi.Transform(
                p=np_to_vec3(min_jerk(init_ee_pos, goal_ee_pos, t, self._T)),
                r=slerp_quat(init_ee_transform.r, goal_ee_transform.r, t, self._T),
            )
            for t in range(self._T)
        ]

    @property
    def horizon(self):
        return self._T

    def __call__(self, scene, env_idx, t_step, t_sim):
        target_transform = self._traj[min(t_step, self._T - 1)]
        tau = self._ee_impedance_ctrlr.compute_tau(env_idx, target_transform)
        self._franka.apply_torque(env_idx, self._franka_name, tau)
