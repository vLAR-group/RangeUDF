import trimesh
import igl
import numpy as np
import os
import traceback
import gc


# number of distance field samples generated per object
sample_num = 100000
cfg = None
def boundary_sampling(sigma, path):
    try:
        file_name = os.path.splitext(os.path.basename(path))[0]
        out_path = path
        input_file = os.path.join(out_path)
        label_file = os.path.join(out_path.replace('scaled.off', 'labels.npz'))
        face_label_map = np.load(label_file)['labels']
        out_file = os.path.dirname(path).replace('data', 'color_data') + '/boundary_{}_samples.npz'.format(sigma)

        if os.path.exists(out_file):
            print('Exists: {}'.format(out_file))
            return

        mesh = trimesh.load(input_file, process=False, maintain_order=True)
        faces = mesh.faces
        vertices = mesh.vertices
        
        points = mesh.sample(sample_num)
        if sigma == 0:
            boundary_points = points
        else:
            boundary_points = points + sigma * np.random.randn(sample_num, 3)

        if sigma == 0:
            df = np.zeros(boundary_points.shape[0])
        else:
            sdf, face_idx, _ = igl.signed_distance(boundary_points, vertices, faces)
            df = np.abs(sdf)

        labels = face_label_map[face_idx]  # get corresponding face label for each point
        np.savez(out_file, points=boundary_points, df = df, labels=labels)
        print('Off-surface sampling finished: {}'.format(out_file))

    except:
        print('Error with {}: {}'.format(path, traceback.format_exc()))


    del mesh, df, boundary_points, labels, points
    gc.collect()


def init(cfg_param):
    global cfg
    cfg = cfg_param
