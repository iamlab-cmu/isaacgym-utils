import numpy as np
import quaternion

from carbongym import gymapi
from carbongym_utils.assets import GymFranka, GymBoxAsset, GymURDFAsset
from carbongym_utils.math_utils import np_to_quat, np_to_vec3, transform_to_np_rpy, rpy_to_quat, transform_to_np
from carbongym_utils.ctrl_utils import ForcePositionController, MovingMedianFilter

from gym.spaces import Box

from .vec_env import GymVecEnv


class GymFrankaVecEnv(GymVecEnv):

    _ACTUATION_MODE_MAP = {
                            'vic': 'attractors',
                            'hfpc': 'torques',
                            'hfpc_cartesian_gains': 'torques',
                            'joints': 'joints'
                        }

    def _fill_scene(self, cfg):
        self._table = GymBoxAsset(self._scene.gym, self._scene.sim, **cfg['table']['dims'],
                            shape_props=cfg['table']['shape_props'],
                            asset_options=cfg['table']['asset_options']
                            )
        self._actuation_mode = self._ACTUATION_MODE_MAP[cfg['franka']['action']['mode']]
        franka = GymFranka(cfg['franka'], self._scene.gym, self._scene.sim,
                            actuation_mode=self._actuation_mode)
        self._frankas = [franka] * self.n_envs

        table_pose = gymapi.Transform(
            p=gymapi.Vec3(cfg['table']['dims']['width']/3, cfg['table']['dims']['height']/2, 0)
        )
        self._franka_pose = gymapi.Transform(
            p=gymapi.Vec3(0, cfg['table']['dims']['height'] + 0.01, 0),
            r=rpy_to_quat([np.pi/2, np.pi/2, -np.pi/2])
        )

        self._franka_name = 'franka0'
        self._table_name = 'table0'
        self._scene.add_asset(self._table_name, self._table, table_pose)
        self._scene.add_asset(self._franka_name, franka, self._franka_pose, collision_filter=2) # avoid self-collisions

    def _init_action_space(self, cfg):
        action_cfg = cfg['franka']['action'][cfg['franka']['action']['mode']]
        self._action_mode = cfg['franka']['action']['mode']

        if self._action_mode == 'vic':
            limits_low = np.array(
                [-action_cfg['max_tra_delta']] * 3 + \
                [-np.deg2rad(action_cfg['max_rot_delta'])] * 3 + \
                [action_cfg['min_stiffness']])
            limits_high = np.array(
                [action_cfg['max_tra_delta']] * 3 + \
                [np.deg2rad(action_cfg['max_rot_delta'])] * 3 + \
                [action_cfg['max_stiffness']])

            self._init_ee_transforms = [
                self._frankas[env_idx].get_desired_ee_transform(env_idx, self._franka_name)
                for env_idx in range(self.n_envs)
            ]
        elif self._action_mode == 'hfpc':
            self._force_filter = MovingMedianFilter(6, 3)
            self._fp_ctrlr = ForcePositionController(
                np.zeros(6), np.zeros(6), np.ones(6), 7)
            limits_low = np.array(
                        [-action_cfg['max_tra_delta']] * 3 + \
                        [-np.deg2rad(action_cfg['max_rot_delta'])] * 3 + \
                        [-np.deg2rad(action_cfg['max_force_delta'])] * 3 + \
                        [0] * 3 + \
                        [action_cfg['min_pos_kp'], action_cfg['min_force_kp']])
            limits_high = np.array(
                        [action_cfg['max_tra_delta']] * 3 + \
                        [np.deg2rad(action_cfg['max_rot_delta'])] * 3 + \
                        [np.deg2rad(action_cfg['max_force_delta'])] * 3 + \
                        [1] * 3 + \
                        [action_cfg['max_pos_kp'], action_cfg['max_force_kp']])
        elif self._action_mode == 'hfpc_cartesian_gains':
            self._force_filter = MovingMedianFilter(6, 3)
            self._fp_ctrlr = ForcePositionController(
                np.zeros(6), np.zeros(6), np.ones(6), 7,
                use_joint_gains_for_position_ctrl=False,
                use_joint_gains_for_force_ctrl=False
            )

            limits_low = np.array(
                        [-action_cfg['max_tra_delta']] * 3 + \
                        [-np.deg2rad(action_cfg['max_rot_delta'])] * 3 + \
                        [-np.deg2rad(action_cfg['max_force_delta'])] * 3 + \
                        [0] * 3 + \
                        [action_cfg['min_pos_kp'], action_cfg['min_force_kp']])
            limits_high = np.array(
                        [action_cfg['max_tra_delta']] * 3 + \
                        [np.deg2rad(action_cfg['max_rot_delta'])] * 3 + \
                        [np.deg2rad(action_cfg['max_force_delta'])] * 3 + \
                        [1] * 3 + \
                        [action_cfg['max_pos_kp'], action_cfg['max_force_kp']])

        elif self._action_mode == 'joints':
            max_rot_delta = np.deg2rad(action_cfg['max_rot_delta'])
            limits_high = np.array([max_rot_delta] * 7)
            limits_low = -limits_high
        else:
            raise ValueError('Unknown action mode!')

        # gripper action
        limits_low = np.concatenate([limits_low, [0]])
        limits_high = np.concatenate([limits_high, [1]])

        action_space = Box(limits_low, limits_high, dtype=np.float32)
        return action_space

    def _init_obs_space(self, cfg):
        '''
        Observations contains:

        joint angles - 7
        gripper width (0 to 0.08) - 1
        ee position - 3
        ee quat - 4
        ee contact forces - 3
        '''
        limits_low = np.array(
            self._frankas[0].joint_limits_lower.tolist()[:-2] + \
            [0] + \
            [-10] * 3 + \
            [-1] * 4 + \
            [-1e-5] * 3
        )
        limits_high = np.array(
            self._frankas[0].joint_limits_upper.tolist()[:-2] + \
            [0.08] + \
            [10] * 3 + \
            [1] * 4 + \
            [1e-5] * 3
        )
        obs_space = Box(limits_low, limits_high, dtype=np.float32)
        return obs_space

    def _apply_actions(self, all_actions):
        for env_idx, env_ptr in enumerate(self._scene.env_ptrs):
            ah = self._scene.ah_map[env_idx][self._franka_name]

            action = all_actions[env_idx]
            arm_action = action[:-1]

            if self._action_mode == 'vic':
                delta_tra = arm_action[:3]
                delta_rpy = arm_action[3:6]
                stiffness = arm_action[6]

                self._frankas[env_idx].set_attractor_props(env_idx, env_ptr, self._franka_name,
                {
                    'stiffness': stiffness,
                    'damping': 2 * np.sqrt(stiffness)
                })

                delta_transform = gymapi.Transform(
                    p=np_to_vec3(delta_tra),
                    r=rpy_to_quat(delta_rpy),
                )
                self._frankas[env_idx].set_delta_ee_transform(env_ptr, env_idx, self._franka_name, delta_transform)
            elif self._action_mode == 'hfpc' or self._action_mode == 'hfpc_cartesian_gains':
                xa_tf = self._frankas[env_idx].get_ee_transform(env_ptr, self._franka_name)
                xa = transform_to_np(xa_tf, format='wxyz')

                fa = -np.concatenate([self._frankas[env_idx].get_ee_ct_forces(env_ptr, ah),
                                      [0,0,0]], axis=0)
                fa = self._force_filter.step(fa)
                J = self._frankas[env_idx].get_jacobian(env_ptr, self._franka_name)

                # The last two points are finger joints.
                qdot = self._frankas[env_idx].get_joints_velocity(env_ptr, ah)[:7]
                xdot = np.matmul(J, qdot)

                xd_position = xa[:3] + arm_action[:3]
                xd_orient_rpy = arm_action[3:6]
                xd_orient_quat = quaternion.from_euler_angles(xd_orient_rpy)
                xd_orient = quaternion.as_float_array(xd_orient_quat)
                xd = np.concatenate([xd_position, xd_orient])

                fd = np.concatenate([fa[:3] + arm_action[6:9], [0, 0, 0]])
                S = np.concatenate([arm_action[9:12], [1, 1, 1]])

                pos_kp, force_kp = arm_action[12:14]
                pos_kd = 2 * np.sqrt(pos_kp)
                force_ki = 0.01 * force_kp
                self._fp_ctrlr.set_ctrls(force_kp, force_ki, pos_kp, pos_kd)
                self._fp_ctrlr.set_targets(xd=xd, fd=fd, S=S)

                tau = self._fp_ctrlr.step(xa, xdot, fa, J, qdot)
                self._frankas[env_idx].apply_torque(env_ptr, ah, tau)
            elif self._action_mode == 'joints':
                delta_joints = np.concatenate([arm_action, [0, 0]]) # add dummy gripper joint cmds
                self._frankas[env_idx].apply_delta_joint_targets(env_ptr, ah, delta_joints)
            else:
                raise ValueError(f"Invalid action mode: {self._action_mode}")

            gripper_action = action[-1]
            gripper_width = np.clip(gripper_action, 0, 0.04)
            self._frankas[env_idx].set_gripper_width_target(env_ptr, ah, gripper_width)

    def _compute_obs(self, all_actions):
        all_obs = np.zeros((self.n_envs, 18))

        for env_idx, env_ptr in enumerate(self._scene.env_ptrs):
            ah = self._scene.ah_map[env_idx][self._franka_name]

            all_joints = self._frankas[env_idx].get_joints(env_ptr, ah)
            ee_transform = self._frankas[env_idx].get_ee_transform(env_ptr, self._franka_name)
            ee_ct_forces = self._frankas[env_idx].get_ee_ct_forces(env_ptr, ah)

            all_obs[env_idx, :7] = all_joints[:7]
            all_obs[env_idx, 7] = all_joints[-1] * 2 # gripper width is 2 * each gripper's prismatic length
            all_obs[env_idx, 8:15] = transform_to_np(ee_transform, format='wxyz')
            all_obs[env_idx, 15:18] = ee_ct_forces

        return all_obs

    def _compute_rews(self, all_obs, all_actions):
        return np.zeros(self.n_envs)

    def _compute_dones(self, all_obs, all_actions, all_rews):
        return np.zeros(self.n_envs)

    def _reset(self, env_idxs):
        if not self._has_first_reset:
            self._init_joints = []
            for env_idx in env_idxs:
                env_ptr = self._scene.env_ptrs[env_idx]
                ah = self._scene.ah_map[env_idx][self._franka_name]
                self._init_joints.append(self._frankas[env_idx].get_joints(env_ptr, ah))

        for env_idx in env_idxs:
            env_ptr = self._scene.env_ptrs[env_idx]
            ah = self._scene.ah_map[env_idx][self._franka_name]

            self._frankas[env_idx].set_joints(env_ptr, ah, self._init_joints[env_idx])
            self._frankas[env_idx].set_joints_targets(env_ptr, ah, self._init_joints[env_idx])

            if self._action_mode == 'joints':
                if 'randomize_joints' in self._cfg['franka'] and self._cfg['franka']['randomize_joints']:
                    init_random_joints = np.clip(np.random.normal(self._init_joints[env_idx], \
                        (self._frankas[env_idx].joint_limits_upper - self._frankas[env_idx].joint_limits_lower)/10), self._frankas[env_idx].joint_limits_lower, \
                        self._frankas[env_idx].joint_limits_upper)
                    self._frankas[env_idx].set_joints(env_ptr, ah, init_random_joints)            
                    self._frankas[env_idx].set_joints_targets(env_ptr, ah, init_random_joints)

            if self._action_mode == 'vic':
                self._frankas[env_idx].set_ee_transform(env_ptr, env_idx, self._franka_name, 
                                            self._init_ee_transforms[env_idx])


class GymFrankaBlockVecEnv(GymFrankaVecEnv):

    def _fill_scene(self, cfg):
        super()._fill_scene(cfg)
        self._block = GymBoxAsset(self._scene.gym, self._scene.sim, **cfg['block']['dims'],
                            shape_props=cfg['block']['shape_props'],
                            rb_props=cfg['block']['rb_props'],
                            asset_options=cfg['block']['asset_options']
                            )
        self._banana = GymURDFAsset(
                            cfg['banana']['urdf_path'],
                            self._scene.gym, self._scene.sim,
                            shape_props=cfg['banana']['shape_props'],
                            rb_props=cfg['banana']['rb_props'],
                            asset_options=cfg['banana']['asset_options']
                            )
        self._block_name = 'block0'
        self._banana_name = 'banana0'
        self._scene.add_asset(self._block_name, self._block, gymapi.Transform())
        self._scene.add_asset(self._banana_name, self._banana, gymapi.Transform())

    def _reset(self, env_idxs):
        super()._reset(env_idxs)
        for env_idx in env_idxs:
            env_ptr = self._scene.env_ptrs[env_idx]
            block_ah = self._scene.ah_map[env_idx][self._block_name]
            banana_ah = self._scene.ah_map[env_idx][self._banana_name]

            block_pose = gymapi.Transform(
                p=np_to_vec3(np.array([
                    (np.random.rand()*2 - 1) * 0.1 + 0.5,
                    self._cfg['table']['dims']['height'] + self._cfg['block']['dims']['height'] / 2 + 0.05,
                    (np.random.rand()) * 0.2]))
                )

            banana_pose = gymapi.Transform(
                p=np_to_vec3(np.array([
                    (np.random.rand()*2 - 1) * 0.1 + 0.5,
                    self._cfg['table']['dims']['height'] + 0.05,
                    (-np.random.rand()) * 0.2])),
                r=rpy_to_quat([np.pi/2, np.pi/2, -np.pi/2])
                )

            self._block.set_rb_transforms(env_ptr, block_ah, [block_pose])
            self._banana.set_rb_transforms(env_ptr, banana_ah, [banana_pose])

    def _init_obs_space(self, cfg):
        obs_space = super()._init_obs_space(cfg)

        # add pose of block to obs_space
        limits_low = np.concatenate([
            obs_space.low,
            [-10] * 3 + [-1] * 4
        ])
        limits_high = np.concatenate([
            obs_space.high,
            [10] * 3 + [1] * 4
        ])
        new_obs_space = Box(limits_low, limits_high, dtype=np.float32)

        return new_obs_space

    def _compute_obs(self, all_actions):
        all_obs = super()._compute_obs(all_actions)

        box_pose_obs = np.zeros((self.n_envs, 7))

        for env_idx, env_ptr in enumerate(self._scene.env_ptrs):
            ah = self._scene.ah_map[env_idx][self._block_name]

            block_transform = self._block.get_rb_transforms(env_ptr, ah)[0]
            box_pose_obs[env_idx, :] = transform_to_np(block_transform, format='wxyz')

        all_obs = np.c_[all_obs, box_pose_obs]
        return all_obs
