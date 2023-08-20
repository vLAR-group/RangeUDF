import numpy as np
import trimesh
import os
import traceback
from scipy.spatial import distance as dist


face_label_map, vertex_label_map, cfg = None, None, None
def get_nearest_vertex(point_cloud, vertices, faces):
    ############################
    # point_cloud: (N, 3)
    # vertices: (N, 3, 3)
    # faces: (N, 3)
    ############################

    nearest_vertex_idx = []
    for point, corresponding_vertices, vertex_indices in zip(point_cloud, vertices, faces):
        distances = np.array([dist.euclidean(point, vertex) for vertex in corresponding_vertices])
        nearest = np.argmin(distances)

        vertex_idx = vertex_indices[nearest]
        nearest_vertex_idx.append(vertex_idx)

    return nearest_vertex_idx


def voxelized_pointcloud_sampling(path):
    try:
        for num in num_list:
            scan_name = path.split('/')[-2]
            file_name = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.dirname(path).replace('scans', 'data_color')
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            input_file = os.path.join(out_path.replace('data_color', 'data'),file_name + '_scaled.off')
            out_file = out_path + '/on_surface_{}points.npz'.format(num)

            if os.path.exists(out_file):
                print(f'Exists: {out_file}')
                return

            mesh = trimesh.load(input_file, process=False, maintain_order=True)
            faces = mesh.faces
            vertices = mesh.vertices
            point_cloud, face_idx = mesh.sample(num, return_index=True) # [N, 3] and corresponding face_idx
            all_face_label = face_label_map[scan_name] # [total_face_num, 3]
            all_face_color = face_color_map[scan_name]
            all_vertex_label = vertex_label_map[scan_name]  # [total_vertex_num, 2]

            all_labels = all_face_label[face_idx]  # get corresponding face label for each point
            colors = all_face_color[face_idx]

            ################################################################################################################
            #  1. if the label num of the face > 1, assign points label to its nearest vertex label value.
            #  2. if the label num of the face == 1, assign points label to its face label value.
            ###
            labels = np.zeros((num, 2)) - 1

            # if label_num == 1, use face label
            on_face_id = all_labels[:, 2] == 1
            labels[on_face_id] = all_labels[on_face_id, :2] # assign points to face label

            # if label_num > 1, find nearest neighbour's label value
            edge_points_id = all_labels[:, 2] > 1  # [True, False, True..., ]
            select_faces = faces[face_idx[edge_points_id]] # [num, 3]: [[1, 2, 3], [8, 9 ,10], ...[a, b, c]], the idx of vertices.
            select_vertices = vertices[select_faces] # [num, 3, 3]: coordinates
            nearest_vertex_idx = get_nearest_vertex(point_cloud[edge_points_id], select_vertices, select_faces)
            labels[edge_points_id] = all_vertex_label[nearest_vertex_idx]
            ################################################################################################################
            np.savez(out_file, point_cloud=point_cloud, labels=labels, colors=colors, bb_min = cfg.bb_min, bb_max = cfg.bb_max, res = cfg.input_res)
            print('On-surface sampling finished: {}'.format(out_file))

    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))

def init(cfg_param):
    global face_label_map, face_color_map, vertex_label_map, cfg,num_list
    cfg = cfg_param
    num_list=[cfg.num_points,5*cfg.num_points]
    label_map = np.load('./color_label_map.npz', allow_pickle=True)
    face_label_map = label_map['face_label_map'].item()
    face_color_map = label_map['face_color_map'].item()
    vertex_label_map = label_map['vertex_label_map'].item()
