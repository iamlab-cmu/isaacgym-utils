# isaacgym-utils
This repo contains wrappers and utilities for `isaacgym`

Supported version of Isaac Gym: 1.0rc2

Supported `up_axis` mode: `z` (`UP_AXIS_Z` in the Isaac Gym documentation)

## Installation

### Install IsaacGym

Install IsaacGym from [Nvidia](https://developer.nvidia.com/isaac-gym)

This library supports the latest IsaacGym version - `1.0.preview2`.
It will not work with older versions.

### Install isaacgym-utils

Install `isaacgym-utils`:

```bash
git clone git@github.com:iamlab-cmu/isaacgym-utils.git
pip install -e isaacgym-utils
```

By default, only "core" dependencies are installed. To install dependencies needed for optional `isaagym-utils` capabilities, modify the above `pip install` command to indicate the desired optional capability. Core dependencies will also be installed with any optional capabilities.

Optional capability groups:
```
pip install -e isaacgym-utils[group]
```

where `group` is the following:
- `rl`: reinforcement learning (RL)
- `ray`: parallel IsaacGym instances via Ray
- `all`: install all capability groups

Multiple capabilities can be specified:
```
pip install -e isaacgym-utils[ray,rl]
```

Or, install all capabilities:
```
pip install -e isaacgym-utils[all]
```

## Running examples

Examples scripts can be found in `isaacgym-utils/examples/`.
These scripts need to be run at the root level of `isaacgym-utils`:

```bash
cd isaacgym-utils
python examples/run_franka.py
```

Each example script has a corresponding config file in `cfg/` that can be used to change object properties like friction.

### Inverse Kinematics

See `isaacgym_utils/examples/franka_pick_block_ik.py` for an example of using IK for the Franka Panda robot. IK is provided by the [Robotics Toolbox for Python](https://github.com/petercorke/robotics-toolbox-python).

### Running with Ray

[Ray](https://github.com/ray-project/ray) is a fast and simple framework for building and running distributed applications.

Requires the `[ray]` or `[all]` installation of `isaacgym-utils`.

See `isaacgym_utils/examples/franka_pick_block_ray.py` for an example of running multiple `isaacgym` instances in parallel using Ray.

### RL environment

Requires the `[rl]` or `[all]` installation of `isaacgym-utils`.

See `isaacgym_utils/rl/vec_env.py` for the abstract Vec Env base class that is used for RL.
It contains definitions of methods that are expected to be overwritten by a child class for a specific RL environment.

See `isaacgym_utils/rl/franka_vec_env.py` for an example of an RL env with a Franka robot using joint control, variable impedance control, and hybrid force-position control.

See `examples/run_franka_rl_vec_env.py` for an example of running the RL environment, and refer to the corresponding config for changing various aspects of the environment (e.g. in the YAML config, the fields under `franka.action` determine what type of action space is used).

For new tasks and control schemes, you can make a new class that inherits `GymVecEnv` (or `GymFrankaVecEnv` if using the Franka) and overwrite the appropriate methods.

## Loading external objects
To load external meshes, the meshes need to be wrapped in an URDF file.
See `assets/ycb` for some examples.
The script `scripts/mesh_to_urdf.py` can help make these URDFs, but using it is not necessary.
Then, they can be loaded via `GymURDFAsset`.
See `GymFrankaBlockVecEnv._setup_single_env_gen` in `isaacgym_utils/rl/franka_vec_env.py` for an example.

## Tensor API

To use IsaacGym's Tensor API, set `scene->gym->use_gpu_pipeline: True` in the yaml configs.

This switches `isaacgym-utils`' API to use the Tensor API backend, and you can access the tensors directly using `scene.tensors`.

To directly write values into writable tensors (see IsaacGym docs for more details), instead of relying on `isaacgym-utils`' internal implementations, you should:
1. Write to a tensor in `scene.tensors`
2. Call `scene.register_actor_tensor_to_update` to ensure that the writes are committed during the next simulation step.

## Things to Note

* Attractors can't be used if `use_gpu_pipeline: True`
* If using `physx` and not controlling the an actor with joint PD control, you must set `dof_props->stiffness` to have all `0`'s, otherwise IsaacGym's internal PD control is still in effect, even if you're sending torque commands or using attractors. This is not a problem when using the `flex` backend.
* We only support `scene->gym->up_axis: z` - setting the `up_axis` to `x` or `y` will give unexpected behaviors for camera rendering. This is a bug internal to IsaacGym.
* `flex` w/ backend supports getting point-wise contacts. `physx` backend w/ `use_gpu_pipeline: True` and `use_gpu: True` only supports getting body-wise contacts (summing up all point-wise contacts). Other `physx` configurations do not support getting any contact forces.
* `flex` does not support `use_gpu_pipeline: True`

## Citation

If you use `isaacgym-utils` in a publication, please consider citing the repo:

```
@misc{isaacgym-utils,
title = {IsaacGym Utilities},
year = {2021},
note = {Developed by the CMU Intelligent Autonomous Manipulation Lab},
url={https://github.com/iamlab-cmu/isaacgym-utils},
author = {Liang, Jacky},
}
```
