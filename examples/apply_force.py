import argparse

import numpy as np
from autolab_core import YamlConfig

from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymBoxAsset
from isaacgym_utils.draw import draw_transforms, draw_contacts
from isaacgym_utils.math_utils import np_to_vec3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/apply_force.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    scene = GymScene(cfg['scene'])
    block = GymBoxAsset(scene, **cfg['block']['dims'],  shape_props=cfg['block']['shape_props'])
    block_name = 'block'
    
    def setup(scene, _):
        scene.add_asset(block_name, block, gymapi.Transform(p=gymapi.Vec3(0, 0, cfg['block']['dims']['sz']/2)))
    scene.setup_all_envs(setup)

    def custom_draws(scene):
        for env_idx in scene.env_idxs:
            block_transform = block.get_rb_transforms(env_idx, block_name)[0]
            draw_transforms(scene, [env_idx], [block_transform], length=0.1)
        draw_contacts(scene, scene.env_idxs)

    def policy(scene, env_idx, t_step, t_sim):
        force = np_to_vec3([-np.sin(t_sim), 0, 0])

        block_transform = block.get_rb_transforms(env_idx, block_name)[0]
        loc = block_transform.p

        block.apply_force(env_idx, block_name, 'box', force, loc)

    scene.run(policy=policy, custom_draws=custom_draws)
