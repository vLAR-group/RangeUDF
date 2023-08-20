#!/bin/bash

bash dir.sh #If the scene dir exists, ignore this command

python preprocess.py --config ../../configs/preprocess/syn.txt --raw_data_dir source/rooms_04 
python preprocess.py --config ../../configs/preprocess/syn.txt --raw_data_dir source/rooms_05 
python preprocess.py --config ../../configs/preprocess/syn.txt --raw_data_dir source/rooms_06 
python preprocess.py --config ../../configs/preprocess/syn.txt --raw_data_dir source/rooms_07 
python preprocess.py --config ../../configs/preprocess/syn.txt --raw_data_dir source/rooms_08 

