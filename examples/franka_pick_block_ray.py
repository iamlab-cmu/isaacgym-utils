import logging
import argparse
from time import sleep

import numpy as np
import ray
from autolab_core import RigidTransform, YamlConfig

from isaacgym import gymapi
from isaacgym_utils.assets import GymBoxAsset, GymFranka
from isaacgym_utils.draw import draw_transforms
from isaacgym_utils.policy import GraspBlockPolicy
from isaacgym_utils.scene import GymScene


def construct_gym_scene():
    scene = GymScene(cfg['scene'])

    table = GymBoxAsset(
        scene.gym,
        scene.sim,
        **cfg['table']['dims'],
        shape_props=cfg['table']['shape_props'],
        asset_options=cfg['table']['asset_options']
    )
    franka = GymFranka(cfg['franka'], scene, actuation_mode='attractors')
    block = GymBoxAsset(
        scene.gym,
        scene.sim,
        **cfg['block']['dims'],
        shape_props=cfg['block']['shape_props'],
        rb_props=cfg['block']['rb_props'],
        asset_options=cfg['block']['asset_options']
    )

    table_transform = gymapi.Transform(p=gymapi.Vec3(cfg['table']['dims']['sx']/3, 0, cfg['table']['dims']['sz']/2))
    franka_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, cfg['table']['dims']['sz'] + 0.01))
 
    def setup(scene, _):
        scene.add_asset('table', table, table_transform)
        scene.add_asset('franka', franka, franka_transform, collision_filter=1)
        scene.add_asset('block', block, gymapi.Transform())
    scene.setup_all_envs(setup)

    return scene, table, franka, block


def run_grasp_block_policy(block_poses):
    scene, table, franka, block = construct_gym_scene()

    def custom_draws(scene):
        for env_idx in scene.env_idxs:
            ee_transform = franka.get_ee_transform(env_idx  , 'franka')
            desired_ee_transform = franka.get_desired_ee_transform(env_idx, 'franka')

            transforms = [ee_transform, desired_ee_transform]

            draw_transforms(scene, [env_idx], transforms)

    policy = GraspBlockPolicy(franka, 'franka', block, 'block')

    # set block poses
    for env_idx in scene.env_idxs:
        block.set_rb_rigid_transforms(env_idx, 'block', [block_poses[env_idx]])

    policy.reset()
    scene.run(time_horizon=policy.time_horizon, policy=policy, custom_draws=custom_draws)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg', '-c', type=str, default='cfg/franka_pick_block_ray.yaml'
    )
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    # Specify the maximum number of cpus and gpus available to ray.
    ray.init(num_cpus=cfg['ray']['num_cpus'], num_gpus=cfg['ray']['num_gpus'])

    # Wait a little for workers to start.
    sleep(2)

    '''
    ray.remote is a decorator.
    Use it with every function you want to parallelize.
    Specify here the resources available to each function call.
    '''
    fn_decorator = ray.remote(
        num_cpus=cfg['ray']['num_cpus_per_fn'],
        num_gpus=cfg['ray']['num_gpus_per_fn'],
        max_calls=1,
    )

    results = []
    logging.info('Starting parallel execution of 4 scenes.')
    for i in range(cfg['ray']['num_cpus']):
        # sample block poses
        block_poses = [
            RigidTransform(
                translation=[
                    (np.random.rand() * 2 - 1) * 0.1 + 0.5,
                    (np.random.rand() * 2 - 1) * 0.2,
                    cfg['table']['dims']['sz']+ cfg['block']['dims']['sz'] / 2 + 0.1,
                ]
            )
            for _ in range(cfg['scene']['n_envs'])
        ]

        '''
        Call the decorated function with the remote method.
        This immediately returns an ObjectID and executes the task in the background.
        '''
        results.append(fn_decorator(run_grasp_block_policy).remote(block_poses))

    # Get the result of all tasks.
    ray.get(results)

    logging.info('Parallel execution of scenes finished.')
