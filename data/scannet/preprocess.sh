#!/bin/bash
# python get_label.py #Ignore this command if color_label_map.npz is available.
python preprocess.py --config ../../configs/preprocess/scannet.txt --raw_data_dir source/scans #Remove internal to_off operations if .off resources exist.
# python create_split.py 