from abc import ABC, abstractmethod
import numpy as np
from isaacgym_utils.scene import GymScene


class GymVecEnv(ABC):

    def __init__(self, cfg, auto_reset_after_done=True, n_inter_steps=1, inter_step_cb=None):
        self._cfg = cfg
        self._scene = GymScene(cfg['scene'])

        self._fill_scene(cfg)
        self._action_space = self._init_action_space(cfg)
        self._obs_space = self._init_obs_space(cfg)
        self._init_rews(cfg)

        self._scene.step() # needed for physics to initialize correctly
        self._has_first_reset = False

        self._step_counts = np.zeros(self.n_envs)
        # Used for logging/debugging
        self._episode_rewards = np.zeros(self.n_envs)

        self._auto_reset_after_done = auto_reset_after_done
        self._n_inter_steps = n_inter_steps
        self._inter_step_cb = inter_step_cb

    @property
    def n_envs(self):
        return self._scene.n_envs

    @property
    def num_envs(self):
        return self.n_envs

    @property
    def action_space(self):
        return self._action_space

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def step_counts(self):
        return self._step_counts.copy()
    
    @property
    def episode_rewards(self):
        return self._episode_rewards.copy()
    
    @property
    def auto_reset_after_done(self):
        return self._auto_reset_after_done

    @auto_reset_after_done.setter
    def auto_reset_after_done(self, auto_reset_after_done):
        self._auto_reset_after_done = auto_reset_after_done

    @property
    def n_inter_steps(self):
        return self._n_inter_steps

    @abstractmethod
    def _fill_scene(self, cfg):
        pass

    @abstractmethod
    def _init_action_space(self, cfg):
        pass

    @abstractmethod
    def _init_obs_space(self, cfg):
        pass

    def _init_rews(self, cfg):
        pass

    @abstractmethod
    def _apply_actions(self, all_actions):
        pass

    def _apply_inter_actions(self, all_actions, t_inter_step, n_inter_steps):
        pass

    def _inter_step_terminate(self, all_actions, t_inter_step, n_inter_steps):
        return False

    @abstractmethod
    def _compute_obs(self, all_actions, is_reset=False):
        pass

    @abstractmethod
    def _compute_rews(self, all_obs, all_actions):
        pass

    @abstractmethod
    def _compute_dones(self, all_obs, all_actions, all_rews):
        pass

    def _compute_infos(self, all_obs, all_actions, all_rews, all_dones):
        return [{} for _ in range(self.n_envs)]

    @abstractmethod
    def _reset(self, env_idxs):
        pass

    def reset(self, env_idxs=None):
        if not self._has_first_reset or env_idxs is None:
            env_idxs = list(range(self.n_envs))
        
        if len(env_idxs) > 0:
            self._reset(env_idxs)
            self._has_first_reset = True

        self._step_counts[env_idxs] = 0
        self._episode_rewards[env_idxs] = 0
        all_obs = self._compute_obs(None)

        return all_obs

    def step(self, all_actions, n_inter_steps=None, inter_step_cb=None):
        self._apply_actions(all_actions)

        if n_inter_steps is None:
            n_inter_steps = self.n_inter_steps
        for t_inter_step in range(n_inter_steps):
            self._apply_inter_actions(all_actions, t_inter_step, n_inter_steps)
            self._scene.step()

            if inter_step_cb is None:
                inter_step_cb = self._inter_step_cb
            if inter_step_cb is not None:
                inter_step_cb(self, t_inter_step, n_inter_steps)

            terminate = self._inter_step_terminate(all_actions, t_inter_step, n_inter_steps)
            if terminate:
                break
        
        all_obs = self._compute_obs(all_actions)
        all_rews = self._compute_rews(all_obs, all_actions)
        all_dones = self._compute_dones(all_obs, all_actions, all_rews)
        all_infos = self._compute_infos(all_obs, all_actions, all_rews, all_dones)

        self._step_counts += 1
        self._episode_rewards += all_rews

        if self._auto_reset_after_done:
            done_env_idx = np.where(all_dones)[0]
            if len(done_env_idx) > 0:
                new_obs = self.reset(done_env_idx)
                for env_idx in done_env_idx:
                    all_infos[env_idx]['terminal_observation'] = all_obs[env_idx]
            
                all_obs = new_obs 

        return all_obs, all_rews, all_dones, all_infos

    def render(self, custom_draws=None):
        self._scene.render(custom_draws=custom_draws)

    def close(self):
        self._scene.close()
