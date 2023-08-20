
import networks.rangeudf as model
import datasets.dataloader_scenenn as voxelized_data
from networks.generation import Generator
import torch
import config as cfg_loader
import os
import numpy as np
from tqdm import tqdm
import pickle


cfg = cfg_loader.get_config()

device = torch.device("cuda")

dataset = voxelized_data.VoxelizedDataset(cfg,
                                          'test',
                                          pointcloud_samples=cfg.num_points,
                                          data_path=cfg.data_dir,
                                          split_file=cfg.split_file,
                                          batch_size=1,
                                          num_sample_points=cfg.num_sample_points_generation,
                                          num_workers=1,
                                          sample_distribution=cfg.sample_ratio,
                                          sample_sigmas=cfg.sample_std_dev)

net = model.D3F(cfg)
cfg_pth='experiments/{}/{}/test_conf_{}.pkl'.format(cfg.exp_name,cfg.log_dir,cfg.ckpt)
with open(cfg_pth,'wb') as f:
    pickle.dump(cfg,f)
gen = Generator(net, cfg, cfg.exp_name, cfg.log_dir, checkpoint=cfg.ckpt, device=device)

if cfg.log_dir is None:
    out_path = './experiments/{}/evaluation/'.format(cfg.exp_name)
else:
    out_path = './experiments/{}/{}/evaluation/'.format(cfg.exp_name, cfg.log_dir)


def gen_iterator(out_path, dataset, gen_p):
    global gen
    gen = gen_p

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(out_path)

    loader = dataset.get_loader(shuffle=False)

    for i, data in tqdm(enumerate(loader)):
        path = os.path.normpath(data['path'][0])

        print(path)
        room_name = path.split('/')[1]
        scene_name = path.split('/')[-1]
        export_path = out_path + '/{}/'.format(cfg.ckpt) + room_name + '/'

        if os.path.exists(export_path):
            print('Path exists - skip! {}'.format(export_path))
            pass
        else:
            os.makedirs(export_path)

        for num_steps in [7]:

            sparse_point_cloud, dense_point_cloud, duration = gen.generate_point_cloud(data, num_steps)
            np.savez(export_path + '{}_dense_point_cloud_{}'.format(scene_name, num_steps), sparse_point_cloud=sparse_point_cloud, dense_point_cloud=dense_point_cloud,duration=duration)


gen_iterator(out_path, dataset, gen)
