from isaacgym import gymapi
from isaacgym.gymutil import AxesGeometry, WireframeSphereGeometry, draw_lines


force_vector_color = gymapi.Vec3(0.7, 0.2, 0.15)
contact_draw_scale = 0.01
def draw_contacts(scene, env_idxs):
    for env_idx in env_idxs:
        env_ptr = scene.env_ptrs[env_idx]
        scene.gym.draw_env_rigid_contacts(scene.viewer, env_ptr, force_vector_color, contact_draw_scale, False)


def draw_transforms(scene, env_idxs, transforms, length=0.05):
    axes_geom = AxesGeometry(length)
    for env_idx in env_idxs:
        env_ptr = scene.env_ptrs[env_idx]
        for transform in transforms:
            draw_lines(axes_geom, scene.gym, scene.viewer, env_ptr, transform)


def draw_spheres(scene, env_idxs, positions, radius, color=None):
    sphere_geom = WireframeSphereGeometry(radius=radius, color=color)
    for env_idx in env_idxs:
        env_ptr = scene.env_ptrs[env_idx]
        for position in positions:
            draw_lines(sphere_geom, scene.gym, scene.viewer, env_ptr, gymapi.Transform(p=position))
