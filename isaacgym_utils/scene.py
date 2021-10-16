from copy import deepcopy
import numpy as np
from numba import jit
from isaacgym import gymapi, gymtorch
import torch

from .math_utils import np_to_vec3
from .constants import isaacgym_VERSION, quat_real_to_gym_cam


class GymScene:

    def __init__(self, cfg):
        self._gym, self._sim = make_gym(cfg['gym'])
        self._use_gpu_pipeline = self._gym.get_sim_params(self._sim).use_gpu_pipeline
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

        self.env_ptrs = []

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
        self._has_ran_setup = False

    def setup_all_envs(self, setup):
        assert not self._has_ran_setup
        
        while self._current_mutable_env_idx < self.n_envs:
            env_ptr = self.gym.create_env(self.sim, self._env_lower, self._env_upper, self._env_num_per_row)
            self.env_ptrs.append(env_ptr)
            
            setup(self, self._current_mutable_env_idx)

            self._current_mutable_env_idx += 1
            
        if self.use_gpu_pipeline:
            self.gym.prepare_sim(self.sim)
            self._tensors = {
                'root': gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim)),
                'rb_states': gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)),
                'net_cf': gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim)),
                'dof_states': gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
            }
            self._tensors.update({
                'dof_targets': self._tensors['dof_states'][:, 0].clone(),
                'dof_actuation_force': self._tensors['dof_states'][:, 0].clone(),
            })

            self._actor_idxs_to_update = {
                'root': [],
                'dof_states': [],
                'dof_targets': [],
                'dof_actuation_force': []
            }
            self.step()

        for env_idx, assets in self._assets.items():
            for name, asset in assets.items():
                asset._post_create_actor(env_idx, name)

        self._has_ran_setup = True

    @property
    def use_gpu_pipeline(self):
        return self._use_gpu_pipeline

    @property
    def gpu_device(self):
        assert self.use_gpu_pipeline
        return self.tensors['root'].device

    @property
    def tensors(self):
        assert self.use_gpu_pipeline
        return self._tensors

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
        assert not self._has_ran_setup

        env_idx = self._current_mutable_env_idx
        if name in self.ch_map[env_idx]:
            raise ValueError('Camera {} has already been added to env {}!'.format(name, env_idx))
        env_ptr = self.env_ptrs[env_idx]

        # convert to the "gym" frame from the "optical" or "real" camera convention
        tform_gym = deepcopy(transform)
        tform_gym.r = tform_gym.r * quat_real_to_gym_cam

        ch = self.gym.create_camera_sensor(env_ptr, camera.gym_cam_props)
        self.gym.set_camera_transform(ch, env_ptr, tform_gym)
        self.ch_map[env_idx][name] = ch

    def attach_camera(self, name, camera, actor_name, rb_name, offset_transform=None, follow_position_only=False):
        assert not self._has_ran_setup

        if offset_transform is None:
            offset_tform_gym = gymapi.Transform()
        else:
            offset_tform_gym = deepcopy(offset_transform)

        # convert to the "gym" frame from the "optical" or "real" camera convention
        offset_tform_gym.r = offset_tform_gym.r * quat_real_to_gym_cam

        env_idx = self._current_mutable_env_idx
        if name in self.ch_map[env_idx]:
            raise ValueError('Camera {} has already been added to env {}!'.format(name, env_idx))
        env_ptr = self.env_ptrs[env_idx]

        ah = self.ah_map[env_idx][actor_name]
        asset = self.get_asset(actor_name)
        rb_idx = asset.rb_names_map[rb_name]
        rbh = self.gym.get_actor_rigid_body_handle(env_ptr, ah, rb_idx)

        ch = self.gym.create_camera_sensor(env_ptr, camera.gym_cam_props)
        if isaacgym_VERSION == '1.0rc1':
            cam_follow_mode = 0 if follow_position_only else 1
        else:
            # only available in at least 1.0rc2
            cam_follow_mode = gymapi.FOLLOW_POSITION if follow_position_only else gymapi.FOLLOW_TRANSFORM
        self.gym.attach_camera_to_body(ch, env_ptr, rbh, offset_tform_gym, cam_follow_mode)
        self.ch_map[env_idx][name] = ch

    def get_asset(self, name, env_idx=0):
        return self._assets[env_idx][name]

    def add_asset(self, name, asset, poses, collision_filter=0):
        assert not self._has_ran_setup

        env_idx = self._current_mutable_env_idx
        env_ptr = self.env_ptrs[env_idx]

        if name in self._assets[env_idx]:
            raise ValueError('Asset {} has already been added to env {}!'.format(name, env_idx))
        self._assets[env_idx][name] = asset

        if type(poses) == list:
            pose = poses[env_idx]
        else:
            pose = poses

        ah = self.gym.create_actor(env_ptr, asset.GLOBAL_ASSET_CACHE[asset.asset_uid], pose, name, env_idx, collision_filter, self._seg_ids[env_idx])
        self.ah_map[env_idx][name] = ah

        asset.set_shape_props(env_idx, name)
        asset.set_rb_props(env_idx, name)
        asset.set_dof_props(env_idx, name)

        self.seg_id_map[env_idx][name] = self._seg_ids[env_idx]
        self._seg_ids[env_idx] += 1

        # update cts cache size
        self._n_rbs = self.gym.get_sim_rigid_body_count(self.sim)
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

        all_cts = self.gym.get_rigid_contacts(self.sim)

        # filter out invalid cts
        eps = 1e-5
        invalid_mask = (all_cts['env0'] == -1) & \
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

    def _register_actor_tensor_to_update(self, env_idx, name, tensor_name):
        env_ptr = self.env_ptrs[env_idx]
        ah = self.ah_map[env_idx][name]
        actor_idx = self.gym.get_actor_index(env_ptr, ah, gymapi.DOMAIN_SIM)
        self._actor_idxs_to_update[tensor_name].append(actor_idx)

    def step(self):
        if self.use_gpu_pipeline:
            if len(self._actor_idxs_to_update['root']) > 0:
                actor_idxs_th = torch.tensor(self._actor_idxs_to_update['root'], device=self.gpu_device)
                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim, 
                    gymtorch.unwrap_tensor(self.tensors['root']), 
                    gymtorch.unwrap_tensor(actor_idxs_th.int()),
                    len(self._actor_idxs_to_update['root'])
                )

            if len(self._actor_idxs_to_update['dof_states']) > 0:
                actor_idxs_th = torch.tensor(self._actor_idxs_to_update['dof_states'], device=self.gpu_device)
                self.gym.set_dof_state_tensor_indexed(
                    self.sim, 
                    gymtorch.unwrap_tensor(self.tensors['dof_states']), 
                    gymtorch.unwrap_tensor(actor_idxs_th.int()),
                    len(self._actor_idxs_to_update['dof_states'])
                )

            if len(self._actor_idxs_to_update['dof_targets']) > 0:
                actor_idxs_th = torch.tensor(self._actor_idxs_to_update['dof_targets'], device=self.gpu_device)
                self.gym.set_dof_position_target_tensor_indexed(
                    self.sim, 
                    gymtorch.unwrap_tensor(self.tensors['dof_targets']), 
                    gymtorch.unwrap_tensor(actor_idxs_th.int()),
                    len(self._actor_idxs_to_update['dof_targets'])
                )

            # set dof torques

            # set forces

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        if self.use_gpu_pipeline:
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)

            for k in self._actor_idxs_to_update:
                self._actor_idxs_to_update[k] = []

        if self.is_cts_enabled and not self.use_gpu_pipeline:
            self._propagate_asset_cts()

    def render(self, custom_draws=None):
        if self.gui:
            self.gym.clear_lines(self._viewer)

            if custom_draws is not None:
                custom_draws(self)

            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self._viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

    def render_cameras(self):
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

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
        self.gym.destroy_sim(self.sim)
        if self.gui:
            self.gym.destroy_viewer(self._viewer)


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
