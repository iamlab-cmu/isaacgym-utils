import argparse

import numpy as np
from autolab_core import YamlConfig

from isaacgym_utils.rl import GymFrankaBlockVecEnv
from isaacgym_utils.draw import draw_transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/run_franka_rl_vec_env.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    vec_env = GymFrankaBlockVecEnv(cfg)

    def custom_draws(scene):
        franka = scene.get_asset('franka0')
        for env_idx in scene.env_idxs:
            ee_transform = franka.get_ee_transform(env_idx, 'franka0')
            draw_transforms(scene, [env_idx], [ee_transform])

    all_obs = vec_env.reset()
    t = 0
    while True:
        all_actions = np.array([vec_env.action_space.sample() for _ in range(vec_env.n_envs)])
        all_obs, all_rews, all_dones, all_infos = vec_env.step(all_actions)
        vec_env.render(custom_draws=custom_draws)

        t += 1
        if t == 100:
            vec_env.reset()
            t = 0
