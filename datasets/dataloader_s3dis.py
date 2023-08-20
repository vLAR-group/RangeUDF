from __future__ import division
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import traceback
import nearest_neighbors


class VoxelizedDataset(Dataset):


    def __init__(self, opt, mode, pointcloud_samples, data_path, split_file ,
                 batch_size, num_sample_points, num_workers, sample_distribution, sample_sigmas):
        self.opt = opt
        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.path = data_path
        self.split = np.load(split_file)

        self.mode = mode
        self.data = self.split[mode]

        self.num_off_surface_points = num_sample_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_on_surface_points = pointcloud_samples

        self.num_samples = np.rint(self.sample_distribution * num_sample_points).astype(np.uint32)
        print('initializing {} dataset'.format(self.mode))

        # Dict from labels to names
        self.ignore_label = np.sort([-1])
        self.label_to_names = {-1: '<UNK>',
                                    0: 'ceiling',
                                    1: 'floor',
                                    2: 'wall',
                                    3: 'beam',
                                    4: 'column',
                                    5: 'window',
                                    6: 'door',
                                    7: 'table',
                                    8: 'chair',
                                    9: 'sofa',
                                    10: 'bookcase',
                                    11: 'board',
                                    12: 'clutter'}

        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.valid_labels = np.sort([c for c in self.label_values if c not in self.ignore_label])
        self.num_classes = len(self.label_to_names) - len(self.ignore_label)
        self.opt.num_classes = self.num_classes

    def __len__(self):
        return len(self.data)

    def get_train_batch(self, idx):
        try:
            path = self.data[idx]
            input_path = self.path + path
            samples_path = self.path + path
            on_surface_path = input_path + '/on_surface_{}points.npz'.format(self.num_on_surface_points)
            on_surface = np.load(on_surface_path)
            on_surface_points = np.array(on_surface['point_cloud'], dtype=np.float32)
            on_surface_label_path = input_path + '/on_surface_{}labels.npz'.format(self.num_on_surface_points)
            on_surface_labels = np.array(np.load(on_surface_label_path)['full'], dtype=np.float32)

            if self.mode == 'test':
                return {'on_surface_points': np.array(on_surface_points, dtype=np.float32), 'path' : path}

            ############################################################
            # prepare off-surface points from '/boundary_{}_samples.npz'.
            ############################################################
            input_dict = {}
            input_off_surface_points = []
            input_off_surface_df = []
            input_off_surface_labels = []
            for i, num in enumerate(self.num_samples):
                boundary_sample_ppath=samples_path+'/boundary_{}_points_{}.npz'.format(self.sample_sigmas[i],10*self.num_on_surface_points)
                boundary_sample_lpath=samples_path+'/boundary_{}_labels_{}.npz'.format(self.sample_sigmas[i],10*self.num_on_surface_points)
                boundary_sample_labels=np.load(boundary_sample_lpath)['full']
                boundary_samples=np.load(boundary_sample_ppath)
                boundary_sample_points=boundary_samples['points']
                boundary_sample_df=boundary_samples['df']
                
                subsample_indices = torch.randint(0, len(boundary_sample_points), (num,))
                input_off_surface_points.extend(boundary_sample_points[subsample_indices])
                input_off_surface_df.extend(boundary_sample_df[subsample_indices])
                input_off_surface_labels.extend(boundary_sample_labels[subsample_indices])

            num_on_surface = len(on_surface_points)
            num_off_surface = self.num_off_surface_points

            input_off_surface_points = np.array(input_off_surface_points, dtype=np.float32)
            input_on_surface_df = [0 for i in range(num_on_surface)]  # [0, 0, 0, ...], len(input_on_surface_df) = 10000
            df = input_on_surface_df + input_off_surface_df

            assert len(input_off_surface_points) == self.num_off_surface_points
            assert len(input_off_surface_df) == self.num_off_surface_points
            assert len(df) == num_on_surface + num_off_surface


            ############################################################
            # prepare on-surface points for RangeUDF input.
            ############################################################

            if not self.opt.fixed_input:
                permutation = torch.randperm(len(on_surface_points))
                on_surface_points = on_surface_points[permutation]
                on_surface_labels = on_surface_labels[permutation]
            else:
                print('Fixed input order')

            if self.opt.in_dim == 3:
                feature = on_surface_points
            elif self.opt.in_dim == 6:
                colors = on_surface['colors'] / 255
                feature = np.concatenate((on_surface_points, colors[permutation]), axis=-1)

            ############################
            # group semantic branch data
            ############################
            input_sem_interp_idx = []
            if self.mode == 'train':
                semantic_branch_points = on_surface_points
                semantic_branch_labels = on_surface_labels
                input_labels = np.concatenate((on_surface_labels, semantic_branch_labels))
                input_sem_interp_idx.append(nearest_neighbors.knn(on_surface_points, on_surface_points, self.opt.num_interp + 1, omp=True)[:, 1:])
  
            elif self.mode =='val':
                semantic_branch_points = input_off_surface_points
                semantic_branch_labels = input_off_surface_labels
                input_labels = np.concatenate((on_surface_labels, semantic_branch_labels))
                input_sem_interp_idx.append(nearest_neighbors.knn(on_surface_points, input_off_surface_points, self.opt.num_interp, omp=True))
            assert len(input_sem_interp_idx) == 1

            ############################
            # encoder input
            ############################
            input_on_surface_points = []
            input_neighbors = []
            input_pools = []
            input_on_interp_idx = []
            input_off_interp_idx = []
            input_off_interp_idx.append(nearest_neighbors.knn(on_surface_points, input_off_surface_points, self.opt.num_interp, omp=True))
            for i in range(self.opt.num_layers):

                neigh_idx = nearest_neighbors.knn(on_surface_points, on_surface_points, self.opt.num_neighbors, omp=True)
                sub_points = on_surface_points[:len(on_surface_points) // self.opt.sub_sampling_ratio[i]]
                down_sample = neigh_idx[:len(on_surface_points) // self.opt.sub_sampling_ratio[i]]
                on_up_sample = nearest_neighbors.knn(sub_points, on_surface_points, 1, omp=True)

                input_on_surface_points.append(on_surface_points)
                input_neighbors.append(neigh_idx)
                input_pools.append(down_sample)
                input_on_interp_idx.append(on_up_sample)
                on_surface_points = sub_points

            ############################################################
            # prepare input dict.
            ############################################################
            # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
            
            targets = - np.ones(len(input_labels))
            for i, c in enumerate(self.valid_labels):
                targets[input_labels == c] = i

            input_dict['feature'] = np.array(feature, dtype=np.float32)
            input_dict['df'] = np.array(df, dtype=np.float32)
            input_dict['targets'] = np.array(targets, dtype=np.long)

            input_dict['on_surface_points'] = input_on_surface_points
            input_dict['off_surface_points'] = input_off_surface_points
            input_dict['sem_branch_points'] = semantic_branch_points

            input_dict['input_neighbors'] = input_neighbors
            input_dict['input_pools'] = input_pools

            input_dict['on_interp_idx'] = input_on_interp_idx
            input_dict['off_interp_idx'] = input_off_interp_idx
            input_dict['sem_interp_idx'] = input_sem_interp_idx

            input_dict['path'] = path
            return input_dict
        except:
            print('Error with {}: {}'.format(path, traceback.format_exc()))
            raise

       

    def get_test_batch(self, idx):
        try:
            path = self.data[idx]
            input_path = self.path + path

            on_surface_path = input_path + '/on_surface_{}points.npz'.format(self.num_on_surface_points)
            on_surface = np.load(on_surface_path)
            on_surface_points = np.array(on_surface['point_cloud'], dtype=np.float32)

            on_surface_label_path = input_path + '/on_surface_{}labels.npz'.format(self.num_on_surface_points)
            on_surface_labels = np.load(on_surface_label_path)['full']

            ############################################################
            # prepare on-surface points for RangeUDF input.
            ############################################################

            if not self.opt.fixed_input:
                permutation = torch.randperm(len(on_surface_points))
                on_surface_points = on_surface_points[permutation]
                on_surface_labels = on_surface_labels[permutation]
            else:
                print('Fixed input order')

            if self.opt.in_dim == 3:
                feature = on_surface_points
            elif self.opt.in_dim == 6:
                colors = on_surface['colors'] / 255
                feature = np.concatenate((on_surface_points, colors[permutation]), axis=-1)

            input_labels = on_surface_labels

            input_on_surface_points = []
            input_neighbors = []
            input_pools = []
            input_on_interp_idx = []
            input_off_interp_idx =[]
            for i in range(self.opt.num_layers):

                neigh_idx = nearest_neighbors.knn(on_surface_points, on_surface_points, self.opt.num_neighbors, omp=True)
                sub_points = on_surface_points[:len(on_surface_points) // self.opt.sub_sampling_ratio[i]]
                down_sample = neigh_idx[:len(on_surface_points) // self.opt.sub_sampling_ratio[i]]
                on_up_sample = nearest_neighbors.knn(sub_points, on_surface_points, 1, omp=True)

                input_on_surface_points.append(on_surface_points)
                input_neighbors.append(neigh_idx)
                input_pools.append(down_sample)
                input_on_interp_idx.append(on_up_sample)
                on_surface_points = sub_points


            ############################################################
            # prepare input dict.
            ############################################################
            # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
            targets = - np.ones(len(input_labels))
            for i, c in enumerate(self.valid_labels):
                targets[input_labels == c] = i

            input_dict = {}
            input_dict['feature'] = np.array(feature, dtype=np.float32)
            input_dict['targets'] = np.array(targets, dtype=np.long)
            input_dict['on_surface_points'] = input_on_surface_points
            input_dict['input_neighbors'] = input_neighbors
            input_dict['input_pools'] = input_pools
            input_dict['on_interp_idx'] = input_on_interp_idx
            input_dict['off_interp_idx'] = np.array([])
            input_dict['path'] = path

            input_dict['df'] = np.array([])
            input_dict['off_surface_points'] = np.array([])
            input_dict['on_surface_points_idx'] = np.array([])
            input_dict['off_surface_points_idx'] = np.array([])
            return input_dict
        except:
            print('Error with {}: {}'.format(path, traceback.format_exc()))
            raise

        


    def __getitem__(self, idx):
        if self.mode == 'train' or self.mode == 'val':
            input_dict = self.get_train_batch(idx)
        elif self.mode == 'test':
            input_dict = self.get_test_batch(idx)
        else:
            raise ValueError

        return input_dict

    def get_loader(self, shuffle =True):

        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)
