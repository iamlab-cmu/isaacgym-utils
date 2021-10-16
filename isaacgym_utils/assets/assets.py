from abc import ABC
from pathlib import Path

import numpy as np
import torch

from isaacgym import gymapi
from isaacgym_utils.math_utils import np_to_vec3, np_to_quat, transform_to_RigidTransform, RigidTransform_to_transform


class GymAsset(ABC):

    GLOBAL_ASSET_CACHE = {}
    GLOBAL_TEXTURES_CACHE = {}

    def __init__(self, scene, shape_props={}, rb_props={}, dof_props={}, asset_options={}, assets_root=Path('assets')):
        self._scene = scene
        self._shape_props = shape_props
        self._rb_props = rb_props
        self._dof_props = dof_props

        gym_asset_options = gymapi.AssetOptions()
        for key, value in asset_options.items():
            setattr(gym_asset_options, key, value)
        self._gym_asset_options = gym_asset_options

        self._assets_root = assets_root
        self._sim_rb_idxs_map = {}

    def _insert_asset(self, asset_uid, asset):
        self._asset_uid = asset_uid
        self.GLOBAL_ASSET_CACHE[asset_uid] = asset

        self._rb_names = self._scene.gym.get_asset_rigid_body_names(asset)
        self._rb_names_map = {name: i for i, name in enumerate(self._rb_names)}
        self._rb_count = self._scene.gym.get_asset_rigid_body_count(asset)

    def _post_create_actor(self, env_idx, name):
        if env_idx not in self._sim_rb_idxs_map:
            self._sim_rb_idxs_map[env_idx] = {}
        if name not in self._sim_rb_idxs_map[env_idx]:
            self._sim_rb_idxs_map[env_idx][name] = []
        
        env_ptr = self._scene.env_ptrs[env_idx]
        ah = self._scene.ah_map[env_idx][name]
        for i in range(self._rb_count):
            rb_idx = self._scene.gym.get_actor_rigid_body_index(env_ptr, ah, i, gymapi.DOMAIN_SIM)
            self._sim_rb_idxs_map[env_idx][name].append(rb_idx)

    def _set_cts_cache(self, all_cts_cache, all_cts_loc_cache, all_cts_cache_raw, all_n_cts_cache, all_cts_pairs_cache):
        self._all_cts_cache = all_cts_cache
        self._all_cts_loc_cache = all_cts_loc_cache
        self._all_cts_cache_raw = all_cts_cache_raw
        self._all_n_cts_cache = all_n_cts_cache
        self._all_cts_pairs_cache = all_cts_pairs_cache

    @property
    def asset_uid(self):
        return self._asset_uid

    @property
    def rb_names(self):
        return self._rb_names

    @property
    def rb_names_map(self):
        return self._rb_names_map

    @property
    def rb_count(self):
        return self._rb_count

    def get_shape_props(self, env_idx, name):
        env_ptr = self._scene.env_ptrs[env_idx]
        ah = self._scene.ah_map[env_idx][name]
        return self._scene.gym.get_actor_rigid_shape_properties(env_ptr, ah)
        
    def set_shape_props(self, env_idx, name, shape_props=None):
        if shape_props is None:
            shape_props = self._shape_props

        if shape_props:
            gym_shape_props = self.get_shape_props(env_idx, name)

            if isinstance(shape_props, list):
                if len(shape_props) != len(gym_shape_props):
                    raise ValueError('Length of shape props lists must equal {}'.format(len(gym_shape_props)))
            else:
                shape_props = [shape_props] * len(gym_shape_props)

            for i, gym_shape_prop in enumerate(gym_shape_props):
                if isinstance(shape_props[i], gym_shape_prop.__class__):
                    gym_shape_props[i] = shape_props[i]
                else:
                    for key, val in shape_props[i].items():
                        setattr(gym_shape_prop, key, val)
                
            env_ptr = self._scene.env_ptrs[env_idx]
            ah = self._scene.ah_map[env_idx][name]
            self._scene.gym.set_actor_rigid_shape_properties(env_ptr, ah, gym_shape_props)

    def get_rb_props(self, env_idx, name):
        env_ptr = self._scene.env_ptrs[env_idx]
        ah = self._scene.ah_map[env_idx][name]
        return self._scene.gym.get_actor_rigid_body_properties(env_ptr, ah)

    def set_rb_props(self, env_idx, name, rb_props=None):
        if rb_props is None:
            rb_props = self._rb_props
        if rb_props:
            gym_rb_props = self.get_rb_props(env_idx, name)
            modified_rb_props = False

            if isinstance(rb_props, list):
                if len(rb_props) != len(gym_rb_props):
                    raise ValueError('Length of rb props lists must equal {}'.format(len(gym_rb_props)))
            else:
                rb_props = [rb_props] * len(gym_rb_props)

            env_ptr = self._scene.env_ptrs[env_idx]
            ah = self._scene.ah_map[env_idx][name]

            for rb_idx, gym_rb_prop in enumerate(gym_rb_props):
                if 'mass' in rb_props[rb_idx]:
                    mass = rb_props[rb_idx]['mass']        
                    ratio = mass / gym_rb_prop.mass
                    gym_rb_prop.mass = mass
                    gym_rb_prop.inertia.x.x *= ratio
                    gym_rb_prop.inertia.y.y *= ratio
                    gym_rb_prop.inertia.z.z *= ratio
                    if mass > 0:
                        gym_rb_prop.invMass = 1. / mass
                        gym_rb_prop.invInertia.x.x *= 1. / ratio
                        gym_rb_prop.invInertia.y.y *= 1. / ratio
                        gym_rb_prop.invInertia.z.z *= 1. / ratio
                    modified_rb_props = True
                if 'com' in rb_props[rb_idx]:
                    com = rb_props[rb_idx]['com']
                    gym_rb_prop.com = np_to_vec3(com)
                    modified_rb_props = True
                if 'flags' in rb_props[rb_idx]:
                    if rb_props[rb_idx]['flags'] == 'none':
                        gym_rb_prop.flags = gymapi.RIGID_BODY_NONE
                    elif rb_props[rb_idx]['flags'] == 'no_sim':
                        gym_rb_prop.flags = gymapi.RIGID_BODY_DISABLE_SIMULATION
                    elif rb_props[rb_idx]['flags'] == 'no_gravity':
                        gym_rb_prop.flags = gymapi.RIGID_BODY_DISABLE_GRAVITY
                    modified_rb_props = True

                if 'color' in rb_props[rb_idx]:
                    color = rb_props[rb_idx]['color']
                    self._scene.gym.set_rigid_body_color(env_ptr, ah, rb_idx, gymapi.MESH_VISUAL, np_to_vec3(color))

                if 'texture' in rb_props[rb_idx]:
                    # this is needed for the textures to work for some reason...
                    self._scene.gym.set_rigid_body_color(env_ptr, ah, rb_idx, gymapi.MESH_VISUAL, np_to_vec3([1, 1, 1]))
                    # create and set texture
                    if rb_props[rb_idx]['texture'] not in self.GLOBAL_TEXTURES_CACHE:
                        self.GLOBAL_TEXTURES_CACHE[rb_props[rb_idx]['texture']] = self._scene.gym.create_texture_from_file(self._scene.sim, str(self._assets_root / rb_props[rb_idx]['texture']))
                    th = self.GLOBAL_TEXTURES_CACHE[rb_props[rb_idx]['texture']]
                    self._scene.gym.set_rigid_body_texture(env_ptr, ah, rb_idx, gymapi.MESH_VISUAL, th)

            if modified_rb_props:
                self._scene.gym.set_actor_rigid_body_properties(env_ptr, ah, gym_rb_props)

    def get_dof_props(self, env_idx, name):
        env_ptr = self._scene.env_ptrs[env_idx]
        ah = self._scene.ah_map[env_idx][name]
        return self._scene.gym.get_actor_dof_properties(env_ptr, ah)

    def set_dof_props(self, env_idx, name, dof_props=None):
        if dof_props is None:
            dof_props = self._dof_props

        if dof_props:
            gym_dof_props = self.get_dof_props(env_idx, name)

            for key, val in dof_props.items():
                if key == 'driveMode':
                    if type(val[0]) == str:
                        val = [getattr(gymapi, v) for v in val]
                gym_dof_props[key] = val

            env_ptr = self._scene.env_ptrs[env_idx]
            ah = self._scene.ah_map[env_idx][name]
            self._scene.gym.set_actor_dof_properties(env_ptr, ah, gym_dof_props)

    def get_rb_states(self, env_idx, name):
        env_ptr = self._scene.env_ptrs[env_idx]
        ah = self._scene.ah_map[env_idx][name]

        if self._scene.use_gpu_pipeline:
            rb_states_tensor_idxs = [
                    self._scene.gym.get_actor_rigid_body_index(env_ptr, ah, rb_idx, gymapi.DOMAIN_SIM)
                    for rb_idx in range(self.rb_count)
                ]
            rb_states_tensor = self._scene.tensors['rb_states'][rb_states_tensor_idxs].clone()
            # xyzw -> wxyz
            rb_states_tensor[:, [3, 4, 5, 6]] = rb_states_tensor[:, [6, 3, 4, 5]]
            return rb_states_tensor
        else:
            return self._scene.gym.get_actor_rigid_body_states(env_ptr, ah, gymapi.STATE_ALL)
        
    def set_rb_states(self, env_idx, name, rb_states):
        env_ptr = self._scene.env_ptrs[env_idx]
        ah = self._scene.ah_map[env_idx][name]

        if self._scene.use_gpu_pipeline:
            # can only set root rb state. others are read-only
            root_state = rb_states[0].clone()
            # wxyz -> xyzw
            root_state[[3, 4, 5, 6]] = root_state[[4, 5, 6, 3]]

            actor_idx = self._scene.gym.get_actor_index(env_ptr, ah, gymapi.DOMAIN_SIM)
            self._scene.tensors['root'][actor_idx] = root_state
            self._scene._register_actor_tensor_to_update(env_idx, name, 'root')
            return True
        else:

            return self._scene.gym.set_actor_rigid_body_states(env_ptr, ah, rb_states, gymapi.STATE_ALL)

    def get_rb_poses_as_np_array(self, env_idx, name):
        if self._scene.use_gpu_pipeline:
            return self.get_rb_poses(env_idx, name)
        else:
            pos = self.get_rb_states(env_idx, name)['pose']['p']
            rot = self.get_rb_states(env_idx, name)['pose']['r']

            poses_np = np.zeros([len(pos), 7])
            for i, v in enumerate('xyz'):
                poses_np[:, i] = pos[v]

            for i, v in enumerate('wxyz'):
                poses_np[:, i + 3] = rot[v]

            return poses_np

    def get_rb_poses(self, env_idx, name):
        if self._scene.use_gpu_pipeline:
            return self.get_rb_states(env_idx, name)[:, :7].cpu().numpy()
        else:
            return self.get_rb_states(env_idx, name)['pose']

    def get_rb_vels_as_np_array(self, env_idx, name):
        if self._scene.use_gpu_pipeline:
            return self.get_rb_vels(env_idx, name)
        else:
            vel = self.get_rb_states(env_idx, name)['vel']

            vel_np = np.zeros((len(vel), 2, 3))

            for i, v in enumerate('xyz'):
                vel_np[:, 0, i] = vel['linear'][v]
                vel_np[:, 1, i] = vel['angular'][v]

            return vel_np

    def get_rb_vels(self, env_idx, name):
        if self._scene.use_gpu_pipeline:
            return self.get_rb_states(env_idx, name)[:, 7:].cpu().numpy().reshape(2, 3)
        else:
            return self.get_rb_states(env_idx, name)['vel']

    def get_rb_transforms(self, env_idx, name):
        rb_states = self.get_rb_states(env_idx, name)

        transforms = []
        for i in range(len(rb_states)):
            if self._scene.use_gpu_pipeline:
                translation = rb_states[i, :3].cpu().numpy()
                quaternion = rb_states[i, [4, 5, 6, 3]].cpu().numpy()
            else:
                translation = np.array([rb_states['pose']['p'][k][i] for k in 'xyz'])
                quaternion = np.array([rb_states['pose']['r'][k][i] for k in 'xyzw'])

            transforms.append(gymapi.Transform(p=np_to_vec3(translation), r=np_to_quat(quaternion)))

        return transforms

    def get_rb_transform(self, env_idx, name, rb_name):
        env_ptr = self._scene.env_ptrs[env_idx]

        if self._scene.use_gpu_pipeline:
            ah = self._scene.ah_map[env_idx][name]
            rb_idx = self.rb_names_map[rb_name]
            rb_tensor_idx = self._scene.gym.get_actor_rigid_body_index(
                env_ptr, ah, rb_idx, gymapi.DOMAIN_SIM
            )
            rb_state = self._scene.tensors['rb_states'][rb_tensor_idx]
            return gymapi.Transform(p=np_to_vec3(rb_state[:3]), r=np_to_quat(rb_state[3:7]))
        else:
            bh = self._scene.gym.get_rigid_handle(env_ptr, name, rb_name)
            return self._scene.gym.get_rigid_transform(env_ptr, bh)

    def set_rb_transforms(self, env_idx, name, transforms):
        rb_states = self.get_rb_states(env_idx, name)
        
        if self._scene.use_gpu_pipeline:
            rb_states[:, 7:] = 0
            
            for i, transform in enumerate(transforms):
                for j, k in enumerate('xyz'):
                    rb_states[i][j] = getattr(transform.p, k)
                
                for j, k in enumerate('wxyz'):
                    rb_states[i][j + 3] = getattr(transform.r, k)
        else:
            for k in 'xyz':
                rb_states['vel']['linear'][k] = 0
                rb_states['vel']['angular'][k] = 0

            for i, transform in enumerate(transforms):
                for k in 'xyz':
                    rb_states[i]['pose']['p'][k] = getattr(transform.p, k)

                for k in 'wxyz':
                    rb_states[i]['pose']['r'][k] = getattr(transform.r, k)

        self.set_rb_states(env_idx, name, rb_states)

    def get_rb_rigid_transforms(self, env_idx, name):
        transforms = self.get_rb_transforms(env_idx, name)

        rigid_transforms = [
            transform_to_RigidTransform(transform, from_frame=self._rb_names[i], to_frame='world')
            for i, transform in enumerate(transforms)
        ]

        return rigid_transforms

    def set_rb_rigid_transforms(self, env_idx, name, rigid_transforms):
        transforms = [RigidTransform_to_transform(rigid_transform) for rigid_transform in rigid_transforms]
        self.set_rb_transforms(env_idx, name, transforms)

    def get_rb_ct_forces(self, env_idx, name):
        if self._scene.use_gpu_pipeline:
            env_ptr = self._scene.env_ptrs[env_idx]
            ah = self._scene.ah_map[env_idx][name]

            return self._scene.tensors['net_cf'][[
                self._scene.gym.get_actor_rigid_body_index(env_ptr, ah, rb_idx, gymapi.DOMAIN_SIM)
                for rb_idx in range(self.rb_count)
            ]]
        else:
            return self._all_cts_cache[self._sim_rb_idxs_map[env_idx][name]]

    def get_rb_ct_locs(self, env_idx, name):
        assert not self._scene.use_gpu_pipeline
        return self._all_cts_loc_cache[self._sim_rb_idxs_map[env_idx][name]]

    def get_rb_ct_forces_parts(self, env_idx, name):
        assert not self._scene.use_gpu_pipeline
        return self._all_cts_cache_raw[self._sim_rb_idxs_map[env_idx][name], :, 0]

    def get_rb_ct_locs_parts(self, env_idx, name):
        assert not self._scene.use_gpu_pipeline
        return self._all_cts_cache_raw[self._sim_rb_idxs_map[env_idx][name], :, 1]

    def get_rb_n_cts(self, env_idx, name):
        assert not self._scene.use_gpu_pipeline
        return self._all_n_cts_cache[self._sim_rb_idxs_map[env_idx][name]]

    def get_rb_in_ct(self, env_idx, source_asset_name, target_asset, target_asset_names, source_rb_idx=0, target_rb_idx=0):
        assert not self._scene.use_gpu_pipeline
        source_rb_idx = self._sim_rb_idxs_map[env_idx][source_asset_name][source_rb_idx]
        target_rb_idxs = [target_asset._sim_rb_idxs_map[env_idx][target_asset_name][target_rb_idx] for target_asset_name in target_asset_names]

        return self._all_cts_pairs_cache[source_rb_idx, target_rb_idxs]

    def apply_force(self, env_idx, name, rb_name, force, loc):
        env_ptr = self._scene.env_ptrs[env_idx]
        ah = self._scene.ah_map[env_idx][name]
        bh = self._scene.gym.get_actor_rigid_body_index(env_ptr, ah, self.rb_names_map[rb_name], gymapi.DOMAIN_ENV)

        if self._scene.use_gpu_pipeline:
            for i, k in enumerate('xyz'):
                self._scene.tensors['forces'][env_idx, bh, i] = getattr(force, k)
                self._scene.tensors['forces_pos'][env_idx, bh, i] = getattr(loc, k)
            self._scene._register_actor_tensor_to_update(env_idx, name, 'forces')
            return True
        else:
            return self._scene.gym.apply_body_force(env_ptr, bh, force, loc)


class GymURDFAsset(GymAsset):

    def __init__(self, urdf_path, *args, dof_props={}, **kwargs):
        super().__init__(*args, **kwargs)
        asset_uid = self._assets_root / urdf_path

        self._gym_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        gym_asset = self._scene.gym.load_asset(self._scene.sim, str(self._assets_root), urdf_path, self._gym_asset_options)
        
        self._insert_asset(asset_uid, gym_asset)

        self._dof_props = dof_props
        self._n_dofs = self._scene.gym.get_asset_dof_count(gym_asset)

    @property
    def n_dofs(self):
        return self._n_dofs

    def get_dof_states(self, env_idx, name):
        env_ptr = self._scene.env_ptrs[env_idx]
        ah = self._scene.ah_map[env_idx][name]

        if self._scene.use_gpu_pipeline:
            dof_states_tensor_idxs = [
                    self._scene.gym.get_actor_dof_index(env_ptr, ah, dof_idx, gymapi.DOMAIN_SIM)
                    for dof_idx in range(self.n_dofs)
                ]
            return self._scene.tensors['dof_states'][dof_states_tensor_idxs].clone()
        else:
            return self._scene.gym.get_actor_dof_states(env_ptr, ah, gymapi.STATE_ALL).copy()

    def set_dof_states(self, env_idx, name, dof_states):
        env_ptr = self._scene.env_ptrs[env_idx]
        ah = self._scene.ah_map[env_idx][name]

        if self._scene.use_gpu_pipeline:
            dof_states_tensor_idxs = [
                    self._scene.gym.get_actor_dof_index(env_ptr, ah, dof_idx, gymapi.DOMAIN_SIM)
                    for dof_idx in range(self.n_dofs)
                ]
            self._scene.tensors['dof_states'][dof_states_tensor_idxs] = dof_states
            self._scene._register_actor_tensor_to_update(env_idx, name, 'dof_states')
            return True
        else:
            return self._scene.gym.set_actor_dof_states(env_ptr, ah, dof_states, gymapi.STATE_ALL)

    def get_dof_names_map(self, env_idx, name):
        env_ptr = self._scene.env_ptrs[env_idx]
        ah = self._scene.ah_map[env_idx][name]
        return self._scene.gym.get_actor_dof_dict(env_ptr, ah)

    def get_joints(self, env_idx, name):
        dof_states = self.get_dof_states(env_idx, name)
        if self._scene.use_gpu_pipeline:
            return dof_states[:, 0].cpu().numpy()
        else:
            return dof_states['pos']
    
    def get_joints_velocity(self, env_idx, name):
        dof_states = self.get_dof_states(env_idx, name)
        if self._scene.use_gpu_pipeline:
            return dof_states[:, 1].cpu().numpy()
        else:
            return dof_states['vel']

    def set_joints(self, env_idx, name, joints):
        dof_states = self.get_dof_states(env_idx, name)

        if self._scene.use_gpu_pipeline:
            dof_states[:, 0] = torch.from_numpy(joints).type_as(dof_states).to(dof_states.device)
            dof_states[:, 1] = 0
        else:
            dof_states['pos'] = joints
            dof_states['vel'] *= 0
        return self.set_dof_states(env_idx, name, dof_states)

    def set_joints_velocity(self, env_idx, name, joints_velocity):
        dof_states = self.get_dof_states(env_idx, name)

        if self._scene.use_gpu_pipeline:
            dof_states[:, 1] = torch.from_numpy(joints_velocity).type_as(dof_states).to(dof_states.device)
        else:
            dof_states['vel'] = joints_velocity
        return self.set_dof_states(env_idx, name, dof_states)

    def apply_delta_joints(self, env_idx, name, delta_joints):
        joints = self.get_joints(env_idx, name)
        joints += delta_joints
        return self.set_joints(env_idx, name, joints)

    def get_joints_targets(self, env_idx, name):
        env_ptr = self._scene.env_ptrs[env_idx]
        ah = self._scene.ah_map[env_idx][name]

        if self._scene.use_gpu_pipeline:
            return self.get_joints(env_idx, name)
        else:
            return self._scene.gym.get_actor_dof_position_targets(env_ptr, ah)

    def set_joints_targets(self, env_idx, name, joints):
        env_ptr = self._scene.env_ptrs[env_idx]
        ah = self._scene.ah_map[env_idx][name]

        if self._scene.use_gpu_pipeline:
            dof_targets_tensor_idxs = [
                    self._scene.gym.get_actor_dof_index(env_ptr, ah, dof_idx, gymapi.DOMAIN_SIM)
                    for dof_idx in range(self.n_dofs)
                ]
            self._scene.tensors['dof_targets'][dof_targets_tensor_idxs] = torch.from_numpy(joints)\
                                                        .type_as(self._scene.tensors['dof_targets'])\
                                                        .to(self._scene.gpu_device)
            self._scene._register_actor_tensor_to_update(env_idx, name, 'dof_targets')
            return True
        else:
            return self._scene.gym.set_actor_dof_position_targets(env_ptr, ah, joints.astype('float32'))

    def apply_delta_joint_targets(self, env_idx, name, delta_joints):
        dof_targets = self.get_joints(env_idx, name)
        dof_targets += delta_joints

        return self.set_joints_targets(env_idx, name, dof_targets)

    def apply_actor_dof_efforts(self, env_idx, name, tau):
        env_ptr = self._scene.env_ptrs[env_idx]
        ah = self._scene.ah_map[env_idx][name]

        if self._scene.use_gpu_pipeline:
            dof_states_tensor_idxs = [
                    self._scene.gym.get_actor_dof_index(env_ptr, ah, dof_idx, gymapi.DOMAIN_SIM)
                    for dof_idx in range(self.n_dofs)
                ]
            self._scene.tensors['dof_actuation_force'][dof_states_tensor_idxs] = torch.from_numpy(tau).\
                                                        type_as(self._scene.tensors['dof_actuation_force'])\
                                                        .to(self._scene.gpu_device)

            self._scene._register_actor_tensor_to_update(env_idx, name, 'dof_actuation_force')
            return True
        else:
            return self._scene.gym.apply_actor_dof_efforts(env_ptr, ah, tau.astype('float32'))


class GymBoxAsset(GymAsset):

    def __init__(self, *args, sx=1, sy=1, sz=1, **kwargs):
        super().__init__(*args, **kwargs)
        asset_uid = 'box_{}_{}_{}'.format(sx, sy, sz)
        gym_asset = self._scene.gym.create_box(self._scene.sim, sx, sy, sz, self._gym_asset_options)
        self._insert_asset(asset_uid, gym_asset)

        self._sx = sx
        self._sy = sy
        self._sz = sz
    
    @property
    def sx(self):
        return self._sx

    @property
    def sy(self):
        return self._sy

    @property
    def sz(self):
        return self._sz


class GymTetGridAsset(GymAsset):

    def __init__(self, *args, dimx=10, dimy=10, dimz=10, 
                    spacingx=0.1, spacingy=0.1, spacingz=0.1, 
                    density=10, baseFixed=False, topFixed=False, leftFixed=False, rightFixed=False,
                    soft_material_props={}, **kwargs):
        super().__init__(*args, **kwargs)

        keys = [dimx, dimy, dimz, spacingx, spacingy, spacingz, density, baseFixed, topFixed, leftFixed, rightFixed]
        key = ('_{}' * len(keys)).format(*keys)
        asset_uid = 'tetgrid{}'.format(key)
        
        soft_material = gymapi.SoftMaterial()
        for key, val in soft_material_props.items():
            setattr(soft_material, key, val)

        gym_asset = self._scene.gym.create_tet_grid(self._scene.sim, soft_material, dimx, dimy, dimz, 
                            spacingx, spacingy, spacingz, density, baseFixed, topFixed, leftFixed, rightFixed)
        self._insert_asset(asset_uid, gym_asset)


class GymCapsuleAsset(GymAsset):

    def __init__(self, *args, radius=1, width=1, **kwargs):
        super().__init__(*args, **kwargs)
        asset_uid = 'capsule_{}_{}'.format(radius, width)
        gym_asset = self._scene.gym.create_capsule(self._scene.sim, radius, width, self._gym_asset_options)
        self._insert_asset(asset_uid, gym_asset)

        self._radius = radius
        self._width = width

    @property
    def radius(self):
        return self._radius

    @property
    def width(self):
        return self._width


class GymSphereAsset(GymAsset):

    def __init__(self, *args, radius=1, **kwargs):
        super().__init__(*args, **kwargs)
        asset_uid = 'sphere_{}'.format(radius)
        gym_asset = self._scene.gym.create_sphere(self._scene.sim, radius, self._gym_asset_options)
        self._insert_asset(asset_uid, gym_asset)

        self._radius = radius

    @property
    def radius(self):
        return self._radius
