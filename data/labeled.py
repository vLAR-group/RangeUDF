import numpy as np
import os
import time



def get_idx(scan_name):
    sigmas = [0.08, 0.02, 0.003]
    on_suf_list=[sample_num,5*sample_num]
    for sample in on_suf_list:
        #on-surface-points
        on_file = path + '/' + scan_name + '/on_surface_{}labels.npz'.format(sample)
        on_points=path + '/' + scan_name + '/on_surface_{}points.npz'.format(sample)
        if 'syn' in path:
            on_label = np.zeros(sample)
            np.savez(on_file, full=on_label)
        else:
            data=np.load(on_points)
            if 'scene' in path or 'scan' in path:
                on_label =data['labels'][:,0]#SceneNN has sem label and ins label
            else:
                on_label =data['labels']
            np.savez(on_file, full=on_label)

    for sigma in sigmas:
        off_path=path+'/'+scan_name+'/boundary_{}_samples.npz'.format(sigma)
        off_file = path + '/' + scan_name + '/boundary_{}_labels_{}.npz'.format(sigma, str(10*sample_num))
        off_points=path + '/' + scan_name + '/boundary_{}_points_{}.npz'.format(sigma, str(10*sample_num))
        data=np.load(off_path)
        df=data['df']
        points=data['points']
        if 'syn' in path :
            off_sub_label = np.zeros(sample_num*10)
            np.savez(off_file, full=off_sub_label)
            np.savez(off_points,points=points,df=df)
        else:
            if 'scene' in path or 'scan' in path:
                labels=data['labels'][:,0] 
            else:
                labels =data['labels']
            np.savez(off_file,full=labels)
            np.savez(off_points,points=points,df=df)
    print(scan_name)

sample_num=10000 #10000 for train, 50000 for generate
syn_paths = [ 'synthetic/source/rooms_04',
          'synthetic/source/rooms_05',
          'synthetic/source/rooms_06',
          'synthetic/source/rooms_07',
          'synthetic/source/rooms_08']

SceneNN_paths=['scenenn/source/data_color']

twoD3DS_paths=['s3dis/source/color_data/area_1',
            's3dis/source/color_data/area_2',
            's3dis/source/color_data/area_3',
            's3dis/source/color_data/area_4',
            's3dis/source/color_data/area_6']
scannet_paths=['scannet/source/data_color',]

for path in syn_paths:
        for scan_name in os.listdir(path):

            if '.' not in scan_name:
                get_idx(scan_name)
for path in SceneNN_paths:
        for scan_name in os.listdir(path):

            if '.' not in scan_name:
                get_idx(scan_name)
for path in twoD3DS_paths:
        for scan_name in os.listdir(path):

            if '.' not in scan_name:
                get_idx(scan_name)
for path in scannet_paths:
        for scan_name in os.listdir(path):

            if '.' not in scan_name:
                get_idx(scan_name)

