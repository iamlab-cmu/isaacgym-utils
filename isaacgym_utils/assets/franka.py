import numpy as np
from pathlib import Path

# needed for IK
import roboticstoolbox as rtb
from spatialmath import SE3, SO3
from spatialmath.quaternion import UnitQuaternion

from isaacgym import gymapi
from isaacgym_utils.constants import (
    franka_default_joints_for_ik,
    isaacgym_utils_ASSETS_PATH,
)
from isaacgym_utils.math_utils import transform_to_RigidTransform, vec3_to_np, quat_to_rot, np_to_vec3

from .assets import GymURDFAsset
from .franka_numerical_utils import get_franka_mass_matrix


class GymFranka(GymURDFAsset):

    INIT_JOINTS = np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4, 0.04, 0.04])
    _LOWER_LIMITS = None
    _UPPER_LIMITS = None
    _VEL_LIMITS = None

    _URDF_PATH = 'franka_description/robots/franka_panda.urdf'
    _URDF_PATH_WITH_DYNAMICS = 'franka_description/robots/franka_panda_dynamics.urdf'

    @staticmethod
    def _key(env_idx, name):
        return (env_idx, name)

    def __init__(self, cfg, *args, actuation_mode='joints'):
        if 'urdf' in cfg:
            urdf_path = cfg['urdf']
            assets_root = Path(cfg['assets_root'])
        else:
            urdf_path = GymFranka._URDF_PATH_WITH_DYNAMICS
            assets_root = isaacgym_utils_ASSETS_PATH
        super().__init__(urdf_path, *args,
                        shape_props=cfg['shape_props'],
                        dof_props=cfg['dof_props'],
                        asset_options=cfg['asset_options'],
                        assets_root=assets_root
                        )

        self._use_custom_ee = False
        if 'custom_ee_rb_name' in cfg:
            self._use_custom_ee = True
            self._custom_ee_rb_name = cfg['custom_ee_rb_name']

        self._left_finger_rb_name = cfg.get('custom_left_finger_rb_name', 'panda_leftfinger')
        self._right_finger_rb_name = cfg.get('custom_right_finger_rb_name', 'panda_rightfinger')

        self._ee_tool_offset = gymapi.Transform()
        if 'custom_ee_offset' in cfg:
            self._ee_tool_offset = gymapi.Transform((np_to_vec3(cfg['custom_ee_offset'])))

        self._gripper_offset = gymapi.Transform(gymapi.Vec3(0, 0, 0.1034))
        self._finger_offset = gymapi.Transform(gymapi.Vec3(0, 0, 0.045))

        self._actuation_mode = actuation_mode
        self._attractor_handles_map = {}
        self._attractor_transforms_map = {}

        self._attractor_stiffness = cfg['attractor_props']['stiffness']
        self._attractor_damping = cfg['attractor_props']['damping']

        self._robot_model_for_ik = rtb.models.URDF.Panda()

    def set_gripper_width_target(self, env_idx, name, width):
        joints_targets = self.get_joints_targets(env_idx, name)
        joints_targets[-2:] = width
        self.set_joints_targets(env_idx, name, joints_targets)

    def open_grippers(self, env_idx, name):
        self.set_gripper_width_target(env_idx, name, 0.04)

    def close_grippers(self, env_idx, name):
        self.set_gripper_width_target(env_idx, name, 0)

    def set_gripper_width(self, env_idx, name, width):
        width = np.clip(width, self._LOWER_LIMITS[-1], self._UPPER_LIMITS[-1])
        self.set_gripper_width_target(env_idx, name, width)

        joints = self.get_joints(env_idx, name)
        joints[-2] = width
        self.set_joints(env_idx, name, joints)

    def get_gripper_width(self, env_idx, name):
        return self.get_joints(env_idx, name)[-1]

    def get_base_transform(self, env_idx, name):
        env_ptr = self._scene.env_ptrs[env_idx]
        bh = self._scene.gym.get_rigid_handle(env_ptr, name, 'panda_link0')
        base_transform = self._scene.gym.get_rigid_transform(env_ptr, bh)
        return base_transform

    def get_ee_transform(self, env_idx, name, offset=True):
        env_ptr = self._scene.env_ptrs[env_idx]
        bh = self._scene.gym.get_rigid_handle(env_ptr, name, 'panda_hand')
        ee_transform = self._scene.gym.get_rigid_transform(env_ptr, bh)
        if offset:
            ee_transform = ee_transform * self._gripper_offset * self._ee_tool_offset
        return ee_transform

    def get_ee_rigid_transform(self, env_idx, name, offset=True):
        return transform_to_RigidTransform(self.get_ee_transform(env_idx, name, offset=offset),
                                                from_frame='panda_ee', to_frame='panda_link0')

    def get_finger_transforms(self, env_idx, name, offset=True):
        env_ptr = self._scene.env_ptrs[env_idx]
        bh_lf = self._scene.gym.get_rigid_handle(env_ptr, name, self._left_finger_rb_name)
        bh_rf = self._scene.gym.get_rigid_handle(env_ptr, name, self._right_finger_rb_name)
        lf_transform = self._scene.gym.get_rigid_transform(env_ptr, bh_lf)
        rf_transform = self._scene.gym.get_rigid_transform(env_ptr, bh_rf)

        if offset:
            lf_transform = lf_transform * self._finger_offset
            rf_transform = rf_transform * self._finger_offset

        return lf_transform, rf_transform

    def get_desired_ee_transform(self, env_idx, name):
        if self._actuation_mode != 'attractors':
            raise ValueError('Can\'t get desired ee transform when not using attractors!')

        key = self._key(env_idx, name)
        return self._attractor_transforms_map[key]

    def get_left_finger_ct_forces(self, env_idx, name):
        env_ptr = self._scene.env_ptrs[env_idx]
        ah = self._scene.ah_map[env_idx][name]
        rbi_lf = self._scene.gym.get_actor_rigid_body_index(env_ptr, ah, self._rb_names_map[self._left_finger_rb_name], gymapi.DOMAIN_ENV)
        ct_forces_lf = self.get_rb_ct_forces(env_idx, name)[rbi_lf]

        return ct_forces_lf

    def get_right_finger_ct_forces(self, env_idx, name):
        env_ptr = self._scene.env_ptrs[env_idx]
        ah = self._scene.ah_map[env_idx][name]
        rbi_rf = self._scene.gym.get_actor_rigid_body_index(env_ptr, ah, self._rb_names_map[self._right_finger_rb_name], gymapi.DOMAIN_ENV)
        ct_forces_rf = self.get_rb_ct_forces(env_idx, name)[rbi_rf]

        return ct_forces_rf

    def get_ee_ct_forces(self, env_idx, name):
        ah = self._scene.ah_map[env_idx][name]
        if self._use_custom_ee:
            all_ct_forces = self._scene.gym.get_rigid_contact_forces(self._scene.sim)
            env_ptr = self._scene.env_ptrs[env_idx]
            rbi = self._scene.gym.get_actor_rigid_body_index(env_ptr, ah, self._rb_names_map[self._custom_ee_rb_name], gymapi.DOMAIN_SIM)
            ct_forces = np.array([all_ct_forces[rbi][k] for k in 'xyz'])
        else:
            ct_forces_lf = self.get_left_finger_ct_forces(env_idx, name)
            ct_forces_rf = self.get_right_finger_ct_forces(env_idx, name)
            ct_forces = ct_forces_lf + ct_forces_rf

        return ct_forces

    @property
    def joint_limits_lower(self):
        return self._LOWER_LIMITS

    @property
    def joint_limits_upper(self):
        return self._UPPER_LIMITS

    @property
    def joint_max_velocities(self):
        return self._VEL_LIMITS

    def set_actuation_mode(self, mode, env_idx, name):
        self._actuation_mode = mode
        env_ptr = self._scene.env_ptrs[env_idx]
        if self._actuation_mode == 'attractors':
            self.set_dof_props(env_idx, name, {
                'driveMode': [gymapi.DOF_MODE_NONE] * 7 + [gymapi.DOF_MODE_POS] * 2
            })

            key = self._key(env_idx, name)
            if key not in self._attractor_handles_map:
                attractor_props = gymapi.AttractorProperties()
                attractor_props.stiffness = self._attractor_stiffness
                attractor_props.damping= self._attractor_damping
                attractor_props.axes = gymapi.AXIS_ALL

                gripper_handle = self._scene.gym.get_rigid_handle(env_ptr, name, 'panda_hand')
                attractor_props.rigid_handle = gripper_handle
                attractor_props.offset = self._gripper_offset * self._ee_tool_offset

                attractor_handle = self._scene.gym.create_rigid_body_attractor(env_ptr, attractor_props)
                self._attractor_handles_map[key] = attractor_handle

            gripper_transform = self.get_ee_transform(env_idx, name)
            self.set_ee_transform_target(env_idx, name, gripper_transform)
        elif self._actuation_mode == 'joints':
            self.set_dof_props(env_idx, name, {
                'driveMode': [gymapi.DOF_MODE_POS] * 9
            })
        elif self._actuation_mode == 'torques':
            self.set_dof_props(env_idx, name, {
                'driveMode': [gymapi.DOF_MODE_EFFORT] * 7 + [gymapi.DOF_MODE_POS] * 2
            })
        else:
            raise ValueError('Unknown actuation mode! Must be attractors, joints, or torques!')

    def _post_create_actor(self, env_idx, name):
        super()._post_create_actor(env_idx, name)
        self.set_joints(env_idx, name, self.INIT_JOINTS)
        self.set_joints_targets(env_idx, name, self.INIT_JOINTS)

        if self._LOWER_LIMITS is None or self._UPPER_LIMITS is None or self._VEL_LIMITS is None:
            dof_props = self.get_dof_props(env_idx, name)
            self._LOWER_LIMITS = dof_props['lower']
            self._UPPER_LIMITS = dof_props['upper']
            self._VEL_LIMITS = dof_props['velocity']

        self.set_actuation_mode(self._actuation_mode, env_idx, name)

    def set_attractor_props(self, env_idx, name, props):
        if self._actuation_mode != 'attractors':
            raise ValueError('Not using attractors!')
        env_ptr = self._scene.env_ptrs[env_idx]

        key = self._key(env_idx, name)
        ath = self._attractor_handles_map[key]
        attractor_props = self._scene.gym.get_attractor_properties(env_ptr, ath)

        for key, val in props.items():
            setattr(attractor_props, key, val)

        self._scene.gym.set_attractor_properties(env_ptr, ath, attractor_props)

    def set_ee_transform(self, env_idx, name, ee_transform, **set_ee_kwargs):

        # TODO: deal with ee offsets

        ik_solution_found, ik_joint_angles_wo_gripper = self.joints_from_ee_transform_ik(
            env_idx,
            name,
            ee_transform,
            **set_ee_kwargs,
        )

        if ik_solution_found:
            # Set targets for joints/ee etc
            gripper_angles = self.get_joints(env_idx, name)[7:].astype("float")
            new_joint_angles = np.concatenate(
                (
                    ik_joint_angles_wo_gripper,
                    gripper_angles,
                )
            )

            # This should "snap" the robot to the block
            self.set_joints(env_idx, name, new_joint_angles)
            self.set_joints_targets(env_idx, name, new_joint_angles)
            self.set_ee_transform_target(env_idx, name, ee_transform)

        return ik_solution_found

    def joints_from_ee_transform_ik(self, env_idx, name, ee_transform, ik_robot_joints_hint='default'):

        if isinstance(ik_robot_joints_hint, str) and ik_robot_joints_hint == 'default':
            robot_joints_for_ik = franka_default_joints_for_ik
        elif isinstance(ik_robot_joints_hint, str) and ik_robot_joints_hint == 'current':
            robot_joints_for_ik = self.get_joints(env_idx, name)[:7].astype(
                "float"
            )
        else:
            robot_joints_for_ik = ik_robot_joints_hint

        # where is the franka base
        base_tform = self.get_base_transform(env_idx, name)
        desired_ee_pos = ee_transform.p

        desired_ee_pos_wrt_base = vec3_to_np(desired_ee_pos - base_tform.p)

        quat_desired = UnitQuaternion(
            ee_transform.r.w,
            [
                ee_transform.r.x,
                ee_transform.r.y,
                ee_transform.r.z,
            ],
        )

        T = SE3(desired_ee_pos_wrt_base) * SE3(SO3(quat_desired.R))
        sol = self._robot_model_for_ik.ikine_LM(
            T,
            q0=robot_joints_for_ik,
            ilimit=200,
        )

        ik_solution_found = sol.success
        joint_angles = sol.q

        return ik_solution_found, joint_angles

    def set_ee_transform_delta(self, env_idx, name, delta_transform):
        raise NotImplementedError

    def set_ee_transform_target(self, env_idx, name, transform):
        if self._actuation_mode != 'attractors':
            raise ValueError('Can\'t set ee transform when not using attractors!')
        key = self._key(env_idx, name)
        attractor_handle = self._attractor_handles_map[key]

        self._attractor_transforms_map[key] = transform

        env_ptr = self._scene.env_ptrs[env_idx]
        self._scene.gym.set_attractor_target(env_ptr, attractor_handle, transform)

    def set_ee_transform_delta_target(self, env_idx, name, transform):
        ''' This performs delta translation in the global frame and
            delta rotation in the end-effector frame.
        '''
        current_transform = self.get_ee_transform(env_idx, name)
        desired_transform = gymapi.Transform(p=current_transform.p, r=current_transform.r)
        desired_transform.p = desired_transform.p + transform.p
        desired_transform.r = transform.r * desired_transform.r

        self.set_ee_transform_target(env_idx, name, desired_transform)

    def apply_torque(self, env_idx, name, tau):
        if len(tau) == 7:
            tau = np.concatenate([tau, np.zeros(2)])

        self.apply_actor_dof_efforts(env_idx, name, tau)

    def get_links_transforms(self, env_idx, name):
        transforms = []
        env_ptr = self._scene.env_ptrs[env_idx]
        for i in range(1, 8):
            link_name = 'panda_link{}'.format(i)
            bh = self._scene.gym.get_rigid_handle(env_ptr, name, link_name)
            transforms.append(self._scene.gym.get_rigid_transform(env_ptr, bh) )

        return transforms

    def get_links_rigid_transforms(self, env_idx, name):
        transforms = self.get_links_transforms(env_idx, name)
        return [transform_to_RigidTransform(transform,
                                        from_frame='panda_link{}'.format(i+1),
                                        to_frame='panda_link0')
                for i, transform in enumerate(transforms)]

    def get_jacobian(self, env_idx, name, target_joint=7):
        transforms = self.get_links_transforms(env_idx, name)

        if target_joint == 7:
            ee_pos = vec3_to_np(self.get_ee_transform(env_idx, name).p)
        else:
            ee_pos = vec3_to_np(transforms[target_joint].p)

        joints_pos, axes = np.zeros((7, 3)), np.zeros((7, 3))
        for i, transform in enumerate(transforms[:target_joint]):
            joints_pos[i] = vec3_to_np(transform.p)
            axes[i] = quat_to_rot(transform.r)[:, 2]
        J = np.r_[np.cross(axes, ee_pos - joints_pos).T, axes.T]

        return J

    def get_mass_matrix(self, env_idx, name):
        q = self.get_joints(env_idx, name)[:7]
        return get_franka_mass_matrix(q)

    def reset_joints(self, env_idx, name):
        self.set_joints(env_idx, name, self.INIT_JOINTS)
