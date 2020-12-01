from carbongym import gymapi
from carbongym.gymutil import AxesGeometry, WireframeSphereGeometry, draw_lines


force_vector_color = gymapi.Vec3(0.7, 0.2, 0.15)
contact_draw_scale = 0.01
def draw_contacts(gym, viewer, env_ptrs):
    for env_ptr in env_ptrs:
        gym.draw_env_rigid_contacts(viewer, env_ptr, force_vector_color, contact_draw_scale, False)


def draw_transforms(gym, viewer, env_ptrs, transforms, length=0.05):
    axes_geom = AxesGeometry(length)
    for env_ptr in env_ptrs:
        for transform in transforms:
            draw_lines(axes_geom, gym, viewer, env_ptr, transform)


def draw_spheres(gym, viewer, env_ptrs, positions, radius, color=None):
    sphere_geom = WireframeSphereGeometry(radius=radius, color=color)
    for env_ptr in env_ptrs:
        for position in positions:
            draw_lines(sphere_geom, gym, viewer, env_ptr, gymapi.Transform(p=position))
