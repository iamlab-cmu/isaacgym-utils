import os
import random

import numpy as np
from autolab_core import RigidTransform
from numba import jit
import quaternion

from isaacgym import gymapi


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)    
    np.random.seed(seed)


def transform_to_RigidTransform(transform, from_frame='', to_frame=''):
    return RigidTransform(
        rotation=quat_to_rot(transform.r),
        translation=vec3_to_np(transform.p),
        from_frame=from_frame, to_frame=to_frame
    )


def RigidTransform_to_transform(rigid_transform):
    return gymapi.Transform(
            np_to_vec3(rigid_transform.translation), 
            np_to_quat(rigid_transform.quaternion, 'wxyz')
        )


def change_basis(T, R_cb, left=True, right=True):
    if left:
        T = RigidTransform(rotation=R_cb.T, from_frame=T.to_frame, to_frame=T.to_frame) * T
    if right:
        T = T * RigidTransform(rotation=R_cb, from_frame=T.from_frame, to_frame=T.from_frame)
    return T


def vec3_to_np(vec):
    return np.array([vec.x, vec.y, vec.z], dtype=np.float32) if isinstance(vec, gymapi.Vec3) else vec


def quat_to_np(q, format='xyzw'):
    return np.array([getattr(q, k) for k in format], dtype=np.float32)


@jit(nopython=True)
def rot_from_np_quat(q):
    ''' Expects wxyz

    From: https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/jay.htm
    '''
    w, x, y, z = q
    A = np.array([
        [w, -z, y, x],
        [z, w, -x, y],
        [-y, x, w, z],
        [-x, -y, -z, w]
    ])
    B = np.array([
        [w, -z, y, -x],
        [z, w, -x, -y],
        [-y, x, w, -z],
        [x, y, z, w],
    ])
    
    return (A @ B)[:3, :3]


def quat_to_rot(q):
    return rot_from_np_quat(quat_to_np(q, 'wxyz'))


def rpy_to_quat(rpy):
    q = quaternion.from_euler_angles(rpy)
    return gymapi.Quat(q.x, q.y, q.z, q.w)


def quat_to_rpy(q):
    q = quaternion.quaternion(q.w, q.x, q.y, q.z)
    return quaternion.as_euler_angles(q)


def transform_to_np(t, format='xyzw'):
    return np.concatenate([vec3_to_np(t.p), quat_to_np(t.r, format=format)], axis=0)


def transform_to_np_rpy(t):
    T = transform_to_RigidTransform(t)
    return np.concatenate([T.translation, T.euler_angles], axis=0)


def np_to_vec3(ary):
    return gymapi.Vec3(ary[0], ary[1], ary[2])


def np_to_quat(ary, format='xyzw'):
    if format == 'wxyz':
        return gymapi.Quat(ary[1], ary[2], ary[3], ary[0])
    elif format == 'xyzw':
        return gymapi.Quat(ary[0], ary[1], ary[2], ary[3])
    else:
        raise ValueError('Unknown quat format! Must be xyzw or wxyz!')


def rot_to_quat(R):
    return np_to_quat(quaternion.as_float_array(quaternion.from_rotation_matrix(R)), format='wxyz')


def np_quat_to_quat(q):
    return gymapi.Quat(w=q.w, x=q.x, y=q.y, z=q.z)


def np_to_transform(ary, format='xyzw'):
    return gymapi.Transform(p=np_to_vec3(ary[:3]), r=np_to_quat(ary[3:], format=format))


def min_jerk(xi, xf, t, T):
    r = t/T
    return xi + (xf - xi) * (10 * r ** 3 - 15 * r ** 4 + 6 * r ** 5)


def min_jerk_delta(xi, xf, t, T, dt):
    r = t/T
    return (xf - xi) * (30 * r ** 2 - 60 * r ** 3 + 30 * r ** 4) * dt


def slerp_quat(q0, q1, t):
    t = max(min(t, 1), 0)

    qt = quaternion.slerp(
        quaternion.from_float_array(quat_to_np(q0, format='wxyz')), 
        quaternion.from_float_array(quat_to_np(q1, format='wxyz')),
        0, 1, t
    )

    return gymapi.Quat(w=qt.w, x=qt.x, y=qt.y, z=qt.z)


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def rotation_between_axes(v0, v1):
    ''' 
    Calculates R such that R @ v0 = v1
    Uses Rodrigues' formula.
    Assumes v0 and v1 are unit vectors
    '''
    v = np.cross(v0, v1)
    norm_v = np.linalg.norm(v)
    v_axis = v / norm_v

    s = norm_v
    c = v0 @ v1
    
    K = skew(v_axis)
    K2 = K @ K

    R = np.eye(3) + s * K + (1 - c) * K2

    return R


def angle_axis_between_axes(v0, v1):
    ''' Computes the angle-axis rotation that brings v0 to v1
    '''
    v = np.cross(v0, v1)
    norm_v = np.linalg.norm(v)

    if np.isclose(norm_v, 0, atol=1e-5):
        return np.zeros(3)
    
    axis = v / norm_v
    angle = np.arccos(np.clip(v0 @ v1, -1, 1))

    return angle * axis


def rotation_to_angle_axis(R):
    q = quaternion.from_rotation_matrix(R)
    return quaternion.as_rotation_vector(q)


def angle_axis_to_rotation(angle_axis):
    q = quaternion.from_rotation_vector(angle_axis)
    return quaternion.as_rotation_matrix(q)


_R_sim_to_real_franka = np.array([
                                [1, 0, 0], 
                                [0, 0, -1],
                                [0, 1, 0]
                            ])


def real_to_sim_franka_transform(Treal_frame_to_base, Tsim_base_to_world=None):
    Tsim_frame_to_base = change_basis(Treal_frame_to_base, _R_sim_to_real_franka, left=True, right=False)
    if Tsim_base_to_world is None:
        Tsim_frame_to_world = Tsim_frame_to_base.as_frames(Tsim_frame_to_base.from_frame, 'world')
    else:
        Tsim_frame_to_world = Tsim_base_to_world * Tsim_frame_to_base
    return Tsim_frame_to_world


def sim_to_real_franka_transform(Tsim_frame_to_world, Tsim_base_to_world=None):
    if Tsim_base_to_world is None:
        Tsim_frame_to_base = Tsim_frame_to_world
    else:
        Tsim_frame_to_base = Tsim_base_to_world.inverse() * Tsim_frame_to_world

    Treal_frame_to_base = change_basis(Tsim_frame_to_base, _R_sim_to_real_franka.T, left=True, right=False)
    Treal_frame_to_world = Treal_frame_to_base.as_frames(Treal_frame_to_base.from_frame, 'world')
    return Treal_frame_to_world


def angle_axis_between_quats(q0, q1):
    '''
    Finds dq s.t. dq * q1 = q0
    '''
    if quaternion.as_float_array(q1) @ quaternion.as_float_array(q0) < 0:
        q0 = -q0
    dq = q0 * q1.inverse()
    return quaternion.as_rotation_vector(dq)


def compute_task_space_impedance_control(J, curr_transform, target_transform, x_vel, Ks, Ds):
        x_pos = vec3_to_np(curr_transform.p)
        x_quat = quaternion.from_float_array(quat_to_np(curr_transform.r, format='wxyz'))

        xd_pos = vec3_to_np(target_transform.p)
        xd_quat = quaternion.from_float_array(quat_to_np(target_transform.r, format='wxyz'))

        xe_pos = x_pos - xd_pos
        xe_ang_axis = angle_axis_between_quats(x_quat, xd_quat)

        xe = np.concatenate([xe_pos, xe_ang_axis])
        tau = J.T @ (-Ks @ xe - Ds @ x_vel)
        return tau
