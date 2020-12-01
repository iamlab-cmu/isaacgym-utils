import os
import argparse
import logging

from shutil import copyfile


def copy_to_output(src_filename, output_dir):
    copyfile(src_filename, os.path.join(output_dir, os.path.basename(src_filename)))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--urdf_template_path', '-ut', type=str, default='assets/single_obj_urdf_template.urdf')
    parser.add_argument('--data_root', '-d', type=str, default='assets')
    parser.add_argument('--scale', '-s', type=float, default=1.)
    parser.add_argument('--mesh_path', '-m', type=str, required=True)
    parser.add_argument('--col_mesh_path', '-cm', type=str, default='')
    parser.add_argument('--urdf_name', '-u', type=str, required=True)
    parser.add_argument('--visual_origin', '-vo', type=list, default=[0., 0., 0.])
    parser.add_argument('--collision_origin', '-co', type=list, default=[0., 0., 0.])
    parser.add_argument('--mass', '-ma', type=float, default=1.)
    parser.add_argument('--ixx', '-ixx', type=float, default=1e-4)
    parser.add_argument('--iyy', '-iyy', type=float, default=1e-4)
    parser.add_argument('--izz', '-izz', type=float, default=1e-4)
    parser.add_argument('--ixy', '-ixy', type=float, default=0.)
    parser.add_argument('--iyz', '-iyz', type=float, default=0.)
    parser.add_argument('--ixz', '-ixz', type=float, default=0.)
    args = parser.parse_args()

    supported_formats = ['obj']
    mesh_format = os.path.basename(args.mesh_path).split('.')[-1]
    col_mesh_format = os.path.basename(args.col_mesh_path).split('.')[-1] if args.col_mesh_path else mesh_format
    if mesh_format not in supported_formats or col_mesh_format not in supported_formats:
        raise ValueError('Mesh format not supported! Only support {}'.format(supported_formats))

    target_path = os.path.join(args.data_root, args.urdf_name)
    if os.path.isdir(target_path):
        raise ValueError('URDF already exists in data folder!')
    os.makedirs(target_path)

    logging.info('Copying mesh')
    visual_mesh_target_path_rel = os.path.join(args.urdf_name, '{}_visual.{}'.format(args.urdf_name, mesh_format))
    visual_mesh_target_path = os.path.join(args.data_root, visual_mesh_target_path_rel)
    collision_mesh_target_path_rel = os.path.join(args.urdf_name, '{}_collision.{}'.format(args.urdf_name, mesh_format))
    collision_mesh_target_path = os.path.join(args.data_root, collision_mesh_target_path_rel)

    copyfile(args.mesh_path, visual_mesh_target_path)
    if args.col_mesh_path:
        copyfile(args.col_mesh_path, collision_mesh_target_path)

    logging.info('Writing URDF')
    with open(args.urdf_template_path, 'r') as f:
        urdf_template = f.read()

    urdf = urdf_template.replace('$name$', args.urdf_name) \
                        .replace('$visual_origin_xyz$', ' '.join(map(str, args.visual_origin))) \
                        .replace('$visual_geometry_path$', visual_mesh_target_path_rel) \
                        .replace('$collision_origin_xyz$', ' '.join(map(str, args.collision_origin))) \
                        .replace('$collision_geometry_path$', collision_mesh_target_path_rel if args.col_mesh_path else visual_mesh_target_path_rel) \
                        .replace('$mass$', str(args.mass)) \
                        .replace('$ixx$', str(args.ixx)) \
                        .replace('$iyy$', str(args.iyy)) \
                        .replace('$izz$', str(args.izz)) \
                        .replace('$ixy$', str(args.ixy)) \
                        .replace('$iyz$', str(args.iyz)) \
                        .replace('$ixz$', str(args.ixz)) \
                        .replace('$scale$', ' '.join([str(args.scale)]*3))

    with open(os.path.join(target_path, '{}.urdf'.format(args.urdf_name)), 'w') as f:
        f.write(urdf)