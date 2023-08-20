#!/bin/bash

# python mesh_obj.py #Ignore this command if .obj is available.

python preprocess.py --config ../../configs/preprocess/s3dis.txt --raw_data_dir source/data/area_1
python preprocess.py --config ../../configs/preprocess/s3dis.txt --raw_data_dir source/data/area_2
python preprocess.py --config ../../configs/preprocess/s3dis.txt --raw_data_dir source/data/area_3
python preprocess.py --config ../../configs/preprocess/s3dis.txt --raw_data_dir source/data/area_4
python preprocess.py --config ../../configs/preprocess/s3dis.txt --raw_data_dir source/data/area_6

# python split.py #Ignore this command if split_*.npz is available.
