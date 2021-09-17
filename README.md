# isaacgym-utils
This repo contains wrappers and utilities for `isaacgym`

## Installation

### Install IsaacGym

Install IsaacGym from either:
- [Nvidia](https://developer.nvidia.com/isaac-gym)
- IAM Lab's [internal repo](https://github.com/iamlab-cmu/isaacgym.git)

### Install isaacgym-utils

Install `isaacgym-utils`:

```bash
git clone git@github.com:iamlab-cmu/isaacgym-utils.git
pip install -e isaacgym-utils
```

By default, only "core" dependencies are installed. To install dependencies needed for optional `isaagym-utils` capabilities, modify the above `pip install` command to indicate the desired optional capability. The following commands will also install the core dependencies.

Reinforcement learning (RL):
```
pip install -e isaacgym-utils[rl]
```

Parallel IsaacGym instances via Ray:
```
pip install -e isaacgym-utils[ray]
```

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

### Running with Ray

"[Ray](https://github.com/ray-project/ray) is a fast and simple framework for building and running distributed
applications."

Requires the `[ray]` or `[all]` installation of `isaacgym-utils`.

See `isaacgym_utils/examples/franka_pick_block_ray.py` for an example of
running multiple `isaacgym` instances in parallel using Ray.

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
The script `scripts/mesh_to_urdf.py` can help make these URDFs.
Then, they can be loaded via `GymURDFAsset`.
See `GymFrankaBlockVecEnv._file_scene` in `isaacgym_utils/rl/franka_vec_env.py` for an example.
