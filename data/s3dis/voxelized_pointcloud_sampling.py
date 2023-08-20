import numpy as np
import trimesh
import os
import traceback


cfg = None
def voxelized_pointcloud_sampling(path):
    try:
        for num in num_list:
            file_name = os.path.splitext(os.path.basename(path))[0]
            out_path = path
            input_file = os.path.join(out_path)
            label_file = os.path.join(out_path.replace('scaled.off', 'labels.npz'))
            color_file = os.path.join(out_path.replace('scaled.off', 'colors.npz'))
            face_label_map = np.load(label_file)['labels']
            face_color_map = np.load(color_file)['color']
            out_file = os.path.dirname(path).replace('data', 'color_data') + '/on_surface_{}points.npz'.format(num)

            if not os.path.exists(os.path.dirname(path).replace('data', 'color_data')):
                os.makedirs(os.path.dirname(path).replace('data', 'color_data'))

            if os.path.exists(out_file):
                print(f'Exists: {out_file}')
                return

            mesh = trimesh.load(input_file, process=False, maintain_order=True)
            point_cloud, face_idx = mesh.sample(num, return_index=True) # [N, 3] and corresponding face_idx
            labels = face_label_map[face_idx] # [total_face_num, 3]
            colors = face_color_map[face_idx]
            ################################################################################################################
            np.savez(out_file, point_cloud=point_cloud, labels=labels, colors=colors, bb_min = cfg.bb_min, bb_max = cfg.bb_max, res = cfg.input_res)
            print('On-surface sampling finished: {}'.format(out_file))

    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))

def init(cfg_param):
    global cfg, num_list
    cfg = cfg_param
    num_list=[cfg.num_points,5*cfg.num_points]#10k for train, 50k for generation
