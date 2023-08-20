import numpy as np
import os
from pykdtree.kdtree import KDTree
import trimesh
import pickle
import open3d
import typing
import sys
import config as cfg_loader
import multiprocessing as mp
from multiprocessing import Pool, Manager
from functools import partial


def distance_p2p(pointcloud_pred, pointcloud_gt,
                    normals_pred, normals_gt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(pointcloud_gt)
    dist, idx = kdtree.query(pointcloud_pred)

    if normals_pred is None:
        return dist, None

    normals_pred = normals_pred / np.linalg.norm(normals_pred, axis=-1, keepdims=True)
    normals_gt = normals_gt / np.linalg.norm(normals_gt, axis=-1, keepdims=True)

    normals_dot_product = (normals_gt[idx] * normals_pred).sum(axis=-1)
    normals_dot_product = np.abs(normals_dot_product)

    return dist, normals_dot_product


def eval_mesh( mesh_pred, mesh_gt, bb_min, bb_max, n_points=100000):

    pointcloud_pred, idx = mesh_pred.sample(n_points, return_index=True)
    pointcloud_pred = pointcloud_pred.astype(np.float32)
    normals_pred = mesh_pred.face_normals[idx]

    pointcloud_gt, idx = mesh_gt.sample(n_points, return_index=True)
    pointcloud_gt = pointcloud_gt.astype(np.float32)
    normals_gt = mesh_gt.face_normals[idx]

    out_dict = eval_pointcloud(pointcloud_pred, pointcloud_gt, normals_pred, normals_gt)

    """
    bb_len = bb_max - bb_min
    bb_samples = np.random.rand(n_points*10, 3) * bb_len + bb_min

    occ_pred = implicit_waterproofing(mesh_pred, bb_samples)[0]
    occ_gt = implicit_waterproofing(mesh_gt, bb_samples)[0]

    area_union = (occ_pred | occ_gt).astype(np.float32).sum()
    area_intersect = (occ_pred & occ_gt).astype(np.float32).sum()

    out_dict['iou'] =  (area_intersect / area_union)
    """

    return out_dict


def get_threshold_percentage(dist, thresholds):
    ''' Evaluates a point cloud.
    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    '''
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold


def calculate_fscore(gt: open3d.geometry.PointCloud, pr: open3d.geometry.PointCloud, th: float=0.01) -> typing.Tuple[float, float, float]:
    '''Calculates the F-score between two point clouds with the corresponding threshold value.'''
    d1 = open3d.compute_point_cloud_to_point_cloud_distance(gt, pr)
    d2 = open3d.compute_point_cloud_to_point_cloud_distance(pr, gt)
    
    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall+precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0

    return fscore, precision, recall

def eval_pointcloud(pointcloud_pred, pointcloud_gt, thresholds=np.linspace(1./1000, 1, 1000),
                    normals_pred=None, normals_gt=None):

    pointcloud_pred = np.asarray(pointcloud_pred)
    pointcloud_gt = np.asarray(pointcloud_gt)

    # Completeness: how far are the points of the target point cloud
    # from thre predicted point cloud
    completeness, completeness_normals = distance_p2p(
        pointcloud_gt, pointcloud_pred,
        normals_gt, normals_pred
    )
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = completeness ** 2

    completeness = completeness.mean()
    completeness2 = completeness2.mean()


    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(
        pointcloud_pred, pointcloud_gt,
        normals_pred, normals_gt
    )
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = accuracy**2

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()

    # Chamfer distance
    chamfer_l2 = 0.5 * completeness2 + 0.5 * accuracy2
    chamfer_l1 = 0.5 * completeness + 0.5 * accuracy

    F = [
        2 * precision[i] * recall[i] / (precision[i] + recall[i])
        for i in range(len(precision))
    ]

    if not normals_pred is None:
        accuracy_normals = accuracy_normals.mean()
        completeness_normals = completeness_normals.mean()
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
    else:
        accuracy_normals = np.nan
        completeness_normals = np.nan
        normals_correctness = np.nan

    """
    out_dict = {
        'completeness': completeness,
        'accuracy': accuracy,
        'normals completeness': completeness_normals,
        'normals accuracy': accuracy_normals,
        'normals': normals_correctness,
        'completeness2': completeness2,
        'accuracy2': accuracy2,
        'chamfer_l2': chamfer_l2,
        'iou': np.nan
    }

    return out_dict
    """
    return completeness2, accuracy2, chamfer_l1, chamfer_l2, F[4], F[9], F[14], F[19]



def eval(name, dict):
   
    if 'scannet' in cfg.split_file:
        room_name = name
        scan_name = name
        raw_name = cfg.raw_data_dir + '/{}/{}_vh_clean_2_scaled.off'.format(room_name, scan_name, scan_name)
        mesh = trimesh.load(raw_name)
        pointcloud_tgt = mesh.sample(100000)
        dense_pcd = np.load(in_path + '/' +room_name + '/'+scan_name + '_dense_point_cloud_7.npz')['dense_point_cloud']
    elif 'scenenn' in cfg.split_file:
        room_name = name
        scan_name = name
        raw_name = cfg.raw_data_dir + '/{}/{}_scaled.off'.format(room_name, scan_name, scan_name)
        mesh = trimesh.load(raw_name)
        pointcloud_tgt = mesh.sample(100000)
        dense_pcd = np.load(in_path + '/' +room_name + '/'+scan_name + '_dense_point_cloud_7.npz')['dense_point_cloud']
    elif 's3dis' in cfg.split_file:
        room_name = name.split('/')[1]
        scan_name = name.split('/')[2]
        raw_name = cfg.raw_data_dir + '/{}/{}/{}_scaled.off'.format(room_name, scan_name, scan_name)
        mesh = trimesh.load(raw_name)
        pointcloud_tgt = mesh.sample(100000)
        dense_pcd = np.load(in_path + '/' +room_name + '/'+scan_name + '_dense_point_cloud_7.npz')['dense_point_cloud']
    else:
        room_name = name.split('/')[1]
        scan_name = name.split('/')[2]
        raw_name = cfg.raw_data_dir + '/{}/{}/{}.obj'.format(room_name, scan_name.split('_')[0], scan_name.split('_')[0])
        mesh = trimesh.load(raw_name)
        pointcloud_tgt = mesh.sample(100000)
        dense_pcd = np.load(in_path + '/' +room_name + '/'+scan_name + '_dense_point_cloud_7.npz')['dense_point_cloud']



    idx = np.random.choice(len(dense_pcd), 200000, replace=False)#np.random.randint(dense_pcd.shape[0], size=2*100000)
    pointcloud = dense_pcd[idx]

    eps = 0.007
    x_max, x_min = pointcloud_tgt[:, 0].max(), pointcloud_tgt[:, 0].min()
    y_max, y_min = pointcloud_tgt[:, 1].max(), pointcloud_tgt[:, 1].min()
    z_max, z_min = pointcloud_tgt[:, 2].max(), pointcloud_tgt[:, 2].min()

    # add small offsets
    x_max, x_min = x_max + eps, x_min - eps
    y_max, y_min = y_max + eps, y_min - eps
    z_max, z_min = z_max + eps, z_min - eps

    mask_x = (pointcloud[:, 0] <= x_max) & (pointcloud[:, 0] >= x_min)
    mask_y = (pointcloud[:, 1] <= y_max) & (pointcloud[:, 1] >= y_min)
    mask_z = (pointcloud[:, 2] <= z_max) & (pointcloud[:, 2] >= z_min)

    mask = mask_x & mask_y & mask_z
    pointcloud_new = pointcloud[mask]

    # Subsample
    print(scan_name, pointcloud_new.shape)
    # if len(pointcloud_new)<100000:
    #     print(pointcloud_new.shape )
    # idx_new = np.random.randint(len(pointcloud_new),size=10000)
    idx_new= np.random.choice(len(pointcloud_new), 100000, replace=False)
    
    pointcloud = pointcloud_new[idx_new]
    x = eval_pointcloud(pointcloud, pointcloud_tgt)

    dict[raw_name] = x


if __name__ == '__main__':
    cfg = cfg_loader.get_config()
    in_path = 'experiments/{}/{}/evaluation/{}'.format(cfg.exp_name,cfg.log_dir, cfg.ckpt)
    p = Pool(cfg.num_cpus)
    paths = np.load(cfg.split_file)['test']
    paths = sorted(paths)

    return_dict = Manager().dict()
    p.map(partial(eval, dict=return_dict), paths)
    p.close()
    p.join()

    n=0
    list_1=[]
    x_list = []
    l1=[]
    l2=[]
    f1=[]
    f2=[]
    f3=[]
    f4=[]
    l_1=[]
    l_2=[]
    f_1=[]
    f_2=[]
    f_3=[]
    f_4=[]
    fail_list = []
    new_dict = {}
    for scan_name in return_dict.keys():
        x = return_dict[scan_name]
        new_dict[scan_name] = x
        l_1.append(x[2])
        l_2.append(x[3])
        f_1.append(x[4])
        f_2.append(x[5])
        f_3.append(x[6])
        f_4.append(x[7])

        if x[3] > 1e-4:
            n+=1
            fail_list.append(x[3])
            x_list.append(x)
            list_1.append(scan_name)
        else:
            l1.append(x[2])
            l2.append(x[3])
            f1.append(x[4])
            f2.append(x[5])
            f3.append(x[6])
            f4.append(x[7])

    if len(fail_list) != 0:
        print(len(l_1), n)
    else:
        print(len(l_1), n)
    print(np.mean(l_1), np.median(l_1), np.mean(l_2), np.median(l_2), np.mean(f_1), np.mean(f_2), np.mean(f_3), np.mean(f_4))
    a=[np.mean(l_1), np.median(l_1), np.mean(l_2), np.median(l_2), np.mean(f_1), np.mean(f_2), np.mean(f_3), np.mean(f_4)]
    with open('experiments/{}/{}/evaluate_epoch{}.pkl'.format(cfg.exp_name,cfg.log_dir,cfg.ckpt), 'wb') as f:
        pickle.dump(a, f)
