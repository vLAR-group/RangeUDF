from plyfile import PlyData, PlyElement
import numpy as np
import os
import json
import shutil
import csv
import pickle

### nyu40 class
CLASS_LABELS = ['wall','floor','cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds',
                'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refrigerator',
                'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 'nightstand', 'toilet', 'sink', 'lamp',
                'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']
### nyu40 id
CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,18,19,20,21,22,23, 24, 25,26,27, 28,
                      29, 30,31,32, 33, 34, 35, 36, 37,38, 39, 40])

#### modify to your own paths
in_raw_scans_folder = 'source/scenenn_dec24_data'
out_unzip_scans_folder = 'source/data'


def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    # if ints convert
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    return mapping

def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs

def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts

def get_full_3d_mesh_sem_ins_label_nyu40(scene_name=''):
    ####### 3d mesh
    scene_full_3d_mesh = PlyData.read(in_raw_scans_folder + '/' + scene_name + f'/{scene_name}.ply')
    scene_full_3d_pc = np.asarray((scene_full_3d_mesh['vertex'].data).tolist(), dtype=np.float32).reshape([-1, 11])
    scene_full_3d_face = np.asarray((scene_full_3d_mesh['face'].data).tolist(), dtype=np.int32).reshape([-1, 3])

    ####### 3d sem ins
    scene_full_3d_sem_label = scene_full_3d_pc[:, 7]  # 0: unannotated
    scene_full_3d_ins_label = scene_full_3d_pc[:, 6]  # 0: unannotated


    # get faces labels
    face_sem_list = []
    face_ins_list = []
    ins_label_num_list = []
    for pc_indices in scene_full_3d_face:
        pc_sem = list(scene_full_3d_sem_label[pc_indices])
        pc_ins = list(scene_full_3d_ins_label[pc_indices])

        face_sem = max(pc_sem, key=pc_sem.count)
        face_ins = max(pc_ins, key=pc_ins.count)

        face_sem_list.append(face_sem)
        face_ins_list.append(face_ins)
        ins_label_num_list.append(len(np.unique(pc_ins)))

    face_label = np.vstack((face_sem_list, face_ins_list, ins_label_num_list))#1st is sem label，2nd is ins label，list means the several face list 
    vertex_colors = scene_full_3d_pc[:, 8:] # [n, 3]
    face_colors = vertex_colors[scene_full_3d_face].mean(axis=1) # [m, 3]

    return scene_full_3d_pc, scene_full_3d_face, scene_full_3d_sem_label, scene_full_3d_ins_label, face_label, face_colors

def unzip_raw_3d_files(in_raw_scans_folder, out_unzip_scans_folder):
    scene_names = sorted(os.listdir(in_raw_scans_folder))
    face_label_map = {}
    face_color_map = {}
    vertex_label_map = {}
    for scene_name in scene_names:
        print(scene_name)
        scene_full_3d_pc, scene_full_3d_face, scene_full_3d_sem_label, scene_full_3d_ins_label, face_label, face_colors = get_full_3d_mesh_sem_ins_label_nyu40(scene_name)
        face_label_map[scene_name] = face_label.T # (N, 3)
        face_color_map[scene_name] = face_colors
        vertex_label_map[scene_name] = np.vstack((scene_full_3d_sem_label, scene_full_3d_ins_label)).T
        print(face_label_map[scene_name].shape, face_color_map[scene_name].shape, vertex_label_map[scene_name].shape)
        ## to save
    if not os.path.isdir(out_unzip_scans_folder): 
        os.makedirs(out_unzip_scans_folder)
    np.savez_compressed('./color_label_map.npz', face_label_map=face_label_map, face_color_map=face_color_map, vertex_label_map=vertex_label_map)


#############
if __name__ == '__main__':
    unzip_raw_3d_files(in_raw_scans_folder = in_raw_scans_folder, out_unzip_scans_folder=out_unzip_scans_folder)
