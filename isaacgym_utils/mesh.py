import trimesh


def get_mesh_bounding_box(obj_path, scale=1):
    mesh = trimesh.load(obj_path)
    mesh.apply_scale(scale)
    
    bounds = mesh.bounding_box.bounds
    corners = trimesh.bounds.corners(bounds)
    return bounds, corners