import copy
import numpy as np
from numba import jit
from isaacgym import gymapi

from .math_utils import np_to_vec3
from .constants import isaacgym_VERSION, quat_real_to_gym_cam


class GymScene:

    def __init__(self, cfg):
        self._gym, self._sim = make_gym(cfg['gym'])
        self._n_envs = cfg['n_envs']
        self._gui = cfg['gui']
        self._dt = cfg['gym']['dt']
        self._cts = cfg.get('cts', True)

        # gui
        if self._gui:
            self._viewer = self._gym.create_viewer(self._sim, gymapi.CameraProperties())
            if 'cam' in cfg:
                cam_pos = np_to_vec3(cfg['cam']['cam_pos'])
                look_at = np_to_vec3(cfg['cam']['look_at'])
            else:
                cam_pos = gymapi.Vec3(1.5, 1.5, 0)
                look_at = gymapi.Vec3(0.5, 1, 0)
            self._gym.viewer_camera_look_at(self._viewer, None, cam_pos, look_at)

        # create envs
        sim_up_axis = self._gym.get_sim_params(self._sim).up_axis
        if isaacgym_VERSION == '1.0rc1' or sim_up_axis == gymapi.UP_AXIS_Y:
            # In 1.0rc1, the 2nd element is always vertical, no matter the up axis setting
            # this logic retains that behavior
            self._env_lower = gymapi.Vec3(-cfg['es'], 0.0, -cfg['es'])
        else:
            # versions above 1.0rc1 when in UP_AXIS_Z mode
            self._env_lower = gymapi.Vec3(-cfg['es'], -cfg['es'], 0.0)
        self._env_upper = gymapi.Vec3(cfg['es'], cfg['es'], cfg['es'])
        self._env_num_per_row = int(np.sqrt(self._n_envs))

        self.env_ptrs = [self._gym.create_env(self._sim, self._env_lower, self._env_upper, self._env_num_per_row)]

        # track assets
        self._assets = {idx : {} for idx in self.env_idxs}
        self.ah_map = {idx : {} for idx in self.env_idxs}

        # track cameras
        self.ch_map = {idx : {} for idx in self.env_idxs}

        # track segmentations
        self._seg_ids = [1 for _ in range(self._n_envs)]
        self.seg_id_map = {idx : {} for idx in self.env_idxs}

        # track contacts
        self._all_cts_cache = None
        self._all_cts_loc_cache = None
        self._all_n_cts_cache = None
        self._all_cts_cache_raw = None
        self._all_cts_rb_counts_cache = None
        self._all_cts_pairs_cache = None
        self._all_cts_cache_raw_max_cts_per_rb = 20 # this will be dynamically resized

        # current mutable env
        self._current_mutable_env_idx = 0

    @property
    def current_mutable_env_idx(self):
        return self._current_mutable_env_idx

    def increment_mutable_env_idx(self):
        if self._current_mutable_env_idx < self.n_envs - 1:
            self._current_mutable_env_idx += 1
            self.env_ptrs.append(self._gym.create_env(self._sim, self._env_lower, self._env_upper, self._env_num_per_row))
        else:
            raise ValueError('Cannot increment mutable env idx beyond n_envs specified in scene cfg!')

    def setup_all_envs(self, setup):
        while True:
            setup(self, self.current_mutable_env_idx)
            if self.current_mutable_env_idx == self.n_envs - 1:
                break
            self.increment_mutable_env_idx()

    @property
    def dt(self):
        return self._dt

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def env_idxs(self):
        return range(self.n_envs)

    @property
    def gui(self):
        return self._gui

    @property
    def gym(self):
        return self._gym

    @property
    def sim(self):
        return self._sim

    @property
    def viewer(self):
        return self._viewer

    def enable_cts(self):
        self._cts = True

    def disable_cts(self):
        self._cts = False

    @property
    def is_cts_enabled(self):
        return self._cts

    def add_standalone_camera(self, name, camera, transform):
        env_idx = self.current_mutable_env_idx
        if name in self.ch_map[env_idx]:
            raise ValueError('Camera {} has already been added to env {}!'.format(name, env_idx))
        env_ptr = self.env_ptrs[env_idx]

        # convert to the "gym" frame from the "optical" or "real" camera convention
        tform_gym = copy.deepcopy(transform)
        tform_gym.r = tform_gym.r * quat_real_to_gym_cam

        ch = self._gym.create_camera_sensor(env_ptr, camera.gym_cam_props)
        self._gym.set_camera_transform(ch, env_ptr, tform_gym)
        self.ch_map[env_idx][name] = ch

    def attach_camera(self, name, camera, actor_name, rb_name, offset_transform=None, follow_position_only=False):
        if offset_transform is None:
            offset_tform_gym = gymapi.Transform()
        else:
            offset_tform_gym = copy.deepcopy(offset_transform)

        # convert to the "gym" frame from the "optical" or "real" camera convention
        offset_tform_gym.r = offset_tform_gym.r * quat_real_to_gym_cam

        env_idx = self.current_mutable_env_idx
        if name in self.ch_map[env_idx]:
            raise ValueError('Camera {} has already been added to env {}!'.format(name, env_idx))
        env_ptr = self.env_ptrs[env_idx]

        ah = self.ah_map[env_idx][actor_name]
        asset = self.get_asset(actor_name)
        rb_idx = asset.rb_names_map[rb_name]
        rbh = self._gym.get_actor_rigid_body_handle(env_ptr, ah, rb_idx)

        ch = self._gym.create_camera_sensor(env_ptr, camera.gym_cam_props)
        if isaacgym_VERSION == '1.0rc1':
            cam_follow_mode = 0 if follow_position_only else 1
        else:
            # only available in at least 1.0rc2
            cam_follow_mode = gymapi.FOLLOW_POSITION if follow_position_only else gymapi.FOLLOW_TRANSFORM
        self._gym.attach_camera_to_body(ch, env_ptr, rbh, offset_tform_gym, cam_follow_mode)
        self.ch_map[env_idx][name] = ch

    def get_asset(self, name, env_idx=0):
        return self._assets[env_idx][name]

    def add_asset(self, name, asset, poses, collision_filter=0):
        env_idx = self.current_mutable_env_idx
        env_ptr = self.env_ptrs[env_idx]

        if name in self._assets[env_idx]:
            raise ValueError('Asset {} has already been added to env {}!'.format(name, env_idx))
        self._assets[env_idx][name] = asset

        if type(poses) == list:
            pose = poses[env_idx]
        else:
            pose = poses

        ah = self._gym.create_actor(env_ptr, asset.GLOBAL_ASSET_CACHE[asset.asset_uid], pose, name, env_idx, collision_filter, self._seg_ids[env_idx])
        self.ah_map[env_idx][name] = ah

        asset._post_create_actor(env_idx, name)
        asset.set_shape_props(env_idx, name)
        asset.set_rb_props(env_idx, name)
        asset.set_dof_props(env_idx, name)

        self.seg_id_map[env_idx][name] = self._seg_ids[env_idx]
        self._seg_ids[env_idx] += 1

        # update cts cache size
        self._n_rbs = self._gym.get_sim_rigid_body_count(self._sim)
        self._all_cts_cache = np.zeros((self._n_rbs, 3))
        self._all_cts_loc_cache = np.zeros((self._n_rbs, 3))
        self._all_n_cts_cache = np.zeros(self._n_rbs, dtype='int')
        self._all_cts_cache_raw = np.zeros((self._n_rbs, self._all_cts_cache_raw_max_cts_per_rb, 2, 3))
        self._all_cts_rb_counts_cache = np.zeros(self._n_rbs)
        self._all_cts_pairs_cache = np.zeros((self._n_rbs, self._n_rbs), dtype='bool')
        self._update_assets_cts_caches()

    def _update_assets_cts_caches(self):
        for env_idx in self.env_idxs:
            for asset in self._assets[env_idx].values():
                asset._set_cts_cache(self._all_cts_cache, self._all_cts_loc_cache, self._all_cts_cache_raw,
                                    self._all_n_cts_cache, self._all_cts_pairs_cache)

    def _propagate_asset_cts(self):
        self._all_cts_cache[:] = 0
        self._all_cts_loc_cache[:] = 0
        self._all_n_cts_cache[:] = 0
        self._all_cts_cache_raw[:] = 0
        self._all_cts_rb_counts_cache[:] = 0
        self._all_cts_pairs_cache[:] = 0

        all_cts = self._gym.get_rigid_contacts(self._sim)

        # filter out invalid cts
        eps = 1e-5
        invalid_mask = all_cts['env0'] == -1 & \
            np.isclose(all_cts['lambda'], 0, atol=eps) & \
            np.isclose(all_cts['normal']['x'], 0, atol=eps) & \
            np.isclose(all_cts['normal']['y'], 0, atol=eps) & \
            np.isclose(all_cts['normal']['z'], 0, atol=eps)
        all_cts = all_cts[np.logical_not(invalid_mask)]
        if len(all_cts) > 0:
            ct_mags = all_cts['lambda'][:, None]
            ct_dirs = np.c_[all_cts['normal']['x'], all_cts['normal']['y'], all_cts['normal']['z']]
            ct_locs_0 = np.c_[all_cts['localPos0']['x'], all_cts['localPos0']['y'], all_cts['localPos0']['z']]
            ct_locs_1 = np.c_[all_cts['localPos1']['x'], all_cts['localPos1']['y'], all_cts['localPos1']['z']]
            ct_forces = ct_mags * ct_dirs

            np.add.at(self._all_cts_cache, all_cts['body0'], ct_forces)
            np.add.at(self._all_cts_cache, all_cts['body1'], -ct_forces)
            np.add.at(self._all_cts_loc_cache, all_cts['body0'], ct_locs_0)
            np.add.at(self._all_cts_loc_cache, all_cts['body1'], ct_locs_1)
            np.add.at(self._all_n_cts_cache, all_cts['body0'], 1)
            np.add.at(self._all_n_cts_cache, all_cts['body1'], 1)

            non_zero_mask = self._all_n_cts_cache > 0
            self._all_cts_loc_cache[non_zero_mask] /= self._all_n_cts_cache[non_zero_mask][:, None]

            ct_idxs_0 = np.zeros(len(ct_locs_0), dtype='int')
            ct_idxs_1 = np.zeros(len(ct_locs_1), dtype='int')
            _compute_ct_sum_idxs(ct_idxs_0, all_cts['body0'], self._all_cts_rb_counts_cache)
            _compute_ct_sum_idxs(ct_idxs_1, all_cts['body1'], self._all_cts_rb_counts_cache)

            max_n_cts_per_rb = max(ct_idxs_0.max(), ct_idxs_1.max()) + 1 # b/c len = max idx + 1
            if max_n_cts_per_rb > self._all_cts_cache_raw_max_cts_per_rb:
                self._all_cts_cache_raw_max_cts_per_rb = _compute_new_cache_size(max_n_cts_per_rb)
                self._all_cts_cache_raw = np.zeros((self._n_rbs, self._all_cts_cache_raw_max_cts_per_rb, 2, 3))
                self._update_assets_cts_caches()

            self._all_cts_cache_raw[all_cts['body0'], ct_idxs_0, 0, :] = ct_forces
            self._all_cts_cache_raw[all_cts['body0'], ct_idxs_0, 1, :] = ct_locs_0
            self._all_cts_cache_raw[all_cts['body1'], ct_idxs_1, 0, :] = -ct_forces
            self._all_cts_cache_raw[all_cts['body1'], ct_idxs_1, 1, :] = ct_locs_1

            non_plane_ct_mask = (all_cts['body0'] != -1) & (all_cts['body1'] != -1)
            if np.any(non_plane_ct_mask):
                self._all_cts_pairs_cache[all_cts['body0'][non_plane_ct_mask], all_cts['body1'][non_plane_ct_mask]] = True
                self._all_cts_pairs_cache[all_cts['body1'][non_plane_ct_mask], all_cts['body0'][non_plane_ct_mask]] = True

    def step(self):
        self._gym.simulate(self._sim)
        self._gym.fetch_results(self._sim, True)
        if self.is_cts_enabled:
            self._propagate_asset_cts()

    def render(self, custom_draws=None):
        if self.gui:
            self._gym.clear_lines(self._viewer)

            if custom_draws is not None:
                custom_draws(self)

            self._gym.step_graphics(self._sim)
            self._gym.draw_viewer(self._viewer, self._sim, False)
            self._gym.sync_frame_time(self._sim)

    def render_cameras(self):
        self._gym.step_graphics(self._sim)
        self._gym.render_all_camera_sensors(self._sim)

    def run(self, time_horizon=None, policy=None, custom_draws=None, cb=None):
        t_step = 0

        while True:
            t_sim = t_step * self.dt

            if time_horizon is not None and t_step >= time_horizon:
                break

            if policy is not None:
                for env_idx in self.env_idxs:
                    policy(self, env_idx, t_step, t_sim)

            self.step()
            self.render(custom_draws=custom_draws)

            if cb is not None:
                cb(self, t_step, t_sim)

            t_step += 1

    def close(self):
        self._gym.destroy_sim(self._sim)
        if self.gui:
            self._gym.destroy_viewer(self._viewer)


@jit(nopython=True)
def _compute_ct_sum_idxs(ct_idxs, cts_idxs_body, rb_counts):
    for i, rb in enumerate(cts_idxs_body):
        ct_idxs[i] = rb_counts[rb]
        rb_counts[rb] += 1


@jit(nopython=True)
def _compute_new_cache_size(min_size):
    # from https://rushter.com/blog/python-lists-and-tuples/
    new_size = min_size + (min_size // 2 ** 3)
    if min_size < 9:
        new_size += 3
    else:
        new_size += 6

    return new_size


def make_gym(sim_cfg):
    gym = gymapi.acquire_gym()

    physics_engine = gymapi.SIM_FLEX
    sim_params = gymapi.SimParams()
    plane_params = gymapi.PlaneParams()
    compute_device, graphics_device = 0, 0
    for key, val in sim_cfg.items():
        if key == 'type':
            if val == 'flex':
                physics_engine = gymapi.SIM_FLEX
            elif val == 'physx':
                physics_engine = gymapi.SIM_PHYSX
            else:
                raise ValueError('Unkonwn physics engine!')
        elif key in ('flex', 'physx'):
            sim_params_engine = getattr(sim_params, key)
            for engine_key, engine_val in val.items():
                setattr(sim_params_engine, engine_key, engine_val)
            setattr(sim_params, key, sim_params_engine)
        elif key == 'gravity':
            sim_params.gravity = np_to_vec3(val)
        elif key == 'plane':
            for plane_key, plane_val in val.items():
                setattr(plane_params, plane_key, plane_val)
        elif key == 'device':
            compute_device = val['compute']
            graphics_device = val['graphics']
        elif key == 'up_axis':
            if val == 'y':
                continue
            elif val == 'z':
                sim_params.up_axis = gymapi.UP_AXIS_Z
                sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
                plane_params.normal = gymapi.Vec3(0, 0, 1)
            else:
                raise ValueError('Unknown up_axis! Must be y or z')
        else:
            setattr(sim_params, key, val)

    sim = gym.create_sim(compute_device, graphics_device, physics_engine, sim_params)
    gym.add_ground(sim, plane_params)

    return gym, sim
