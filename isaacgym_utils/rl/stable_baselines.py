'''
To make vec_env.GymVecEnv compatible with stable_baselines
'''

from stable_baselines3.common.vec_env import VecEnv
from .franka_vec_env import GymFrankaBlockVecEnv


class StableBaselinesVecEnvAdapter(VecEnv):

    def step_async(self, actions):
        pass

    def step_wait(self):
        pass

    def get_attr(self, attr_name, indices=None):
        pass

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def seed(self, seed):
        pass


class GymFrankaBlockVecEnvStableBaselines(GymFrankaBlockVecEnv, StableBaselinesVecEnvAdapter):
    '''
    An example of how to convert a GymVecEnv to a StableBaselines-compatible VecEnv
    '''

    def __init__(self, *args, **kwargs):
        GymFrankaBlockVecEnv.__init__(self, *args, **kwargs)
