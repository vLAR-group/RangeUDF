import boundary_sampling as boundary_sampling
import voxelized_pointcloud_sampling as voxelized_pointcloud_sampling
from glob import glob
import sys
sys.path.append('../../')
import config as cfg_loader
import multiprocessing as mp
from multiprocessing import Pool
# import partial
from functools import partial
import numpy as np

cfg = cfg_loader.get_config()


print('Finding raw files for preprocessing.')
print(cfg.raw_data_dir + cfg.input_data_glob)
paths = glob(cfg.raw_data_dir + cfg.input_data_glob)
paths = sorted(paths)


if cfg.num_cpus == -1:
	num_cpus = mp.cpu_count()
else:
	num_cpus = cfg.num_cpus

def multiprocess(func):
	p = Pool(num_cpus)
	p.map(func, paths)
	p.close()
	p.join()

print('Start sparse on-surface pointcloud sampling.')
voxelized_pointcloud_sampling.init(cfg)
multiprocess(voxelized_pointcloud_sampling.voxelized_pointcloud_sampling)

print('Start distance field sampling.')
boundary_sampling.init(cfg)
for sigma in cfg.sample_std_dev:
	print(f'Start distance field sampling with sigma: {sigma}.')
	bound=partial(boundary_sampling.boundary_sampling,  sigma)
	multiprocess(bound)
