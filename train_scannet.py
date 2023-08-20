import networks.rangeudf as model
import datasets.dataloader_scannet as voxelized_data
from networks import training
import torch
import config as cfg_loader
import numpy as np
import random
import os
import pickle


def seed_torch(seed=20):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

seed_torch()

cfg = cfg_loader.get_config()
cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_dataset = voxelized_data.VoxelizedDataset(cfg,
                                                'train',
                                                pointcloud_samples=cfg.num_points,
                                                data_path=cfg.data_dir,
                                                split_file=cfg.split_file,
                                                batch_size=cfg.batch_size,
                                                num_sample_points=cfg.num_sample_points_training,
                                                num_workers=30,
                                                sample_distribution=cfg.sample_ratio,
                                                sample_sigmas=cfg.sample_std_dev)

val_dataset = voxelized_data.VoxelizedDataset(cfg,
                                              'val',
                                              pointcloud_samples=cfg.num_points,
                                              data_path=cfg.data_dir,
                                              split_file=cfg.split_file,
                                              batch_size=cfg.batch_size,
                                              num_sample_points=cfg.num_sample_points_training,
                                              num_workers=30,
                                              sample_distribution=cfg.sample_ratio,
                                              sample_sigmas=cfg.sample_std_dev)


net = model.D3F(cfg)
print(net)
if not os.path.exists('experiments/{}/{}'.format(cfg.exp_name,cfg.log_dir)):
    os.makedirs('experiments/{}/{}'.format(cfg.exp_name,cfg.log_dir))
cfg_pth='experiments/{}/{}/tarin_conf.pkl'.format(cfg.exp_name,cfg.log_dir)
with open(cfg_pth,'wb') as f:
    pickle.dump(cfg,f)
trainer = training.Trainer(net,
                                        cfg,
                                        torch.device("cuda"),
                                        train_dataset,
                                        val_dataset,
                                        cfg.exp_name,
                                        cfg.log_dir,
                                        optimizer=cfg.optimizer,
                                        gamma=cfg.gamma,
                                        lr=cfg.lr,
                                        threshold=cfg.max_dist,
                                        checkpoint=cfg.ckpt)

trainer.train_model(cfg.num_epochs)
