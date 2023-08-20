from glob import glob
import ../../config as cfg_loader
import numpy as np
import shutil
import os

cfg = cfg_loader.get_config()


print('Finding raw files for preprocessing.')
paths = glob(cfg.raw_data_dir + '/*.obj')
paths = sorted(paths)
print(paths)

chunks = np.array_split(paths,cfg.num_chunks)
paths = chunks[cfg.current_chunk]


for i in paths:
    obj_name = i.split('.')[0]
    print(obj_name)
    if not os.path.exists(obj_name):
        print(obj_name)
        os.makedirs(obj_name)
    shutil.move(i, obj_name)
