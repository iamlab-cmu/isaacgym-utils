from abc import ABC, abstractmethod
import numpy as np

import quaternion


class PIDController:

    def __init__(self, Kps, Kis=None, Kds=None):
        self._Kps = np.array(Kps)
        self._Kis = Kis if Kis is not None else np.zeros_like(self._Kps)
        self._Kds = Kds if Kds is not None else np.zeros_like(self._Kps)

        self.reset()

    def set_gains(self, Kps=None, Kis=None, Kds=None):
        if Kps is not None:
            self._Kps = Kps
        if Kis is not None:
            self._Kis = Kis
        if Kds is not None:
            self._Kds = Kds

    def step(self, err, err_dot=None):
        if err_dot is None:
            if self._last_err is None:
                err_dot = np.zeros_like(self._Kps)
            else:
                err_dot = err - self._last_err

        ctrl = self._Kps * err + self._Kds * err_dot + self._Kis * self._errs_I

        self._last_err = err
        self._errs_I += err

        return ctrl

    def reset(self):
        self._last_err = None
        self._errs_I = np.zeros_like(self._Kps, dtype=np.float)


class MovingFilter(ABC):

    def __init__(self, dim, window):
        self._buffer = np.zeros((window, dim))
        self.reset()

    def step(self, x):
        self._buffer[self._i] = x

        self._i = (self._i + 1) % len(self._buffer)
        if self._i == len(self._buffer) - 1:
            self._full = True

        N = len(self._buffer) if self._full else self._i
        return self._filter(self._buffer[:N])

    @abstractmethod
    def _filter(self, N):
        pass

    def reset(self):
        self._i = 0
        self._full = False


class MovingAverageFilter(MovingFilter):

    def _filter(self, data):
        return np.mean(data, axis=0)


class MovingMedianFilter(MovingFilter):

    def _filter(self, data):
        return np.median(data, axis=0)


class ForcePositionController:

    def __init__(self, xd, fd, S, n_dof, 
                force_kps=None, force_kis=None,
                pos_kps=None, pos_kds=None,
                use_joint_gains_for_position_ctrl=True,
                use_joint_gains_for_force_ctrl=True):
        self._xd = xd
        self._fd = fd
        self._S = np.diag(S)
        self._Sp = np.eye(6) - self._S

        self._n_dof = n_dof

        self._use_joint_gains_for_position_ctrl = use_joint_gains_for_position_ctrl
        self._use_joint_gains_for_force_ctrl = use_joint_gains_for_force_ctrl

        self._n_force = n_dof if use_joint_gains_for_force_ctrl else 6
        self._n_pos = n_dof if use_joint_gains_for_position_ctrl else 6

        if force_kps is None:
            force_kps = np.ones(self._n_force) * 0.1
        if force_kis is None:
            force_kis = force_kps * 0.1
        if pos_kps is None:
            pos_kps = np.ones(self._n_force) * 40
        if pos_kds is None:
            pos_kds = 2 * np.sqrt(pos_kps)

        self._torque_ctrlr = PIDController(force_kps, force_kis, [0] * self._n_force)
        self._pos_ctrlr = PIDController(pos_kps, [0] * self._n_pos, pos_kds)

    def set_targets(self, xd=None, fd=None, S=None):
        if xd is not None:
            self._xd = xd
        if fd is not None:
            self._fd = fd
        if S is not None:
            if len(S.shape) == 1:
                S = np.diag(S)
            self._S = S
            self._Sp = np.eye(6) - self._S

    def get_K_param_from_input(self, k_param, nd):
        if type(k_param) in (float, int, np.float32, np.int32, np.float64):
            k_param_list = [k_param] * nd
        elif type(k_param) is list:
            assert len(k_param) == nd, "Invalid number of params provided"
            k_param_list = k_param
        else:
            raise ValueError("Invalid force ")
        return np.array(k_param_list)

    def set_ctrls(self, force_kp=None, force_ki=None, pos_kp=None, pos_kd=None):
        force_kp_list = self.get_K_param_from_input(force_kp, self._n_force) \
            if force_kp is not None else None
        force_ki_list = self.get_K_param_from_input(force_ki, self._n_force) \
            if force_ki is not None else None

        pos_kp_list = self.get_K_param_from_input(pos_kp, self._n_pos) \
            if pos_kp is not None else None
        pos_kd_list = self.get_K_param_from_input(pos_kd, self._n_pos) \
            if pos_kd is not None else None

        self._torque_ctrlr.set_gains(Kps=force_kp_list, Kis=force_ki_list)
        self._pos_ctrlr.set_gains(Kps=pos_kp_list, Kis=None, Kds=pos_kd_list)

    def _calculate_position_torque(self, xa, xa_dot, J, qdot):
        # Compute cartesian position error.
        pos_d, orient_d = self._xd[:3], self._xd[3:7]
        pos, orient = xa[:3], xa[3:7]
        xe_pos = pos_d - pos

        orient_d_quat = quaternion.as_quat_array(orient_d)
        orient_quat = quaternion.as_quat_array(orient)
        if orient_d.dot(orient) < 0.0:
            orient = -orient_d

        # This is desired - actual
        error_quat = orient_d_quat * orient_quat.inverse()
        error_angle_axis = error_quat.angle() * quaternion.as_rotation_vector(error_quat)

        xe = np.concatenate([xe_pos, error_angle_axis])
        xes = self._S @ xe
        if self._use_joint_gains_for_position_ctrl:
            q_es = J.T @ xes
            tau_p = self._pos_ctrlr.step(q_es, -qdot)
        else:
            xas_dot = self._S @ xa_dot
            force_p = self._pos_ctrlr.step(xes, -xas_dot)
            tau_p = J.T @ force_p

        return tau_p

    def _calculate_force_torque(self, fa, J):
        fe = self._fd - fa
        fes = self._Sp @ fe
        if self._use_joint_gains_for_force_ctrl:
            tau_es = J.T @ fes
            tau_f = self._torque_ctrlr.step(tau_es)
        else:
            force_f = self._torque_ctrlr.step(fes)
            tau_f = J.T @ force_f

        return tau_f

    def step(self, xa, xa_dot, fa, J, qdot):
        tau_p = self._calculate_position_torque(xa, xa_dot, J, qdot)
        tau_f = self._calculate_force_torque(fa, J)
        tau = tau_p + tau_f

        return tau
