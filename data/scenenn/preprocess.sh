#!/bin/bash
python create_split.py
python get_label.py
python check_label.py #If the file exists, ignore these commands

python preprocess.py --config ../../configs/preprocess/scenenn.txt --raw_data_dir source/scenenn_dec24_data
