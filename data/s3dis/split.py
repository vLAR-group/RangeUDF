import numpy as np
import os

area_list = ['area_1', 'area_2', 'area_3', 'area_4', 'area_6']
x_list = []
y_list = []
z_list = []
for area in area_list:
    split_name = f'.source/split_s3dis_new_{area}'
    copy_list = area_list.copy()
    copy_list.remove(area)
    test_list = os.listdir(f'source/data/{area}')
    train_list = []
    for i in copy_list:
        room_list = os.listdir(f'source/data/{i}')
        for room_name in room_list:
            train_list.append('/' + i + '/' + room_name)

    new_val = []
    for i in test_list:
        if 'UNK' in i:
            continue
        new_val.append('/' + area + '/' + i)

    new_train = []
    for j in train_list:
        if 'UNK' in j:
            continue
        new_train.append(j)

    new_test = []
    for i in test_list:
        if 'UNK' in j:
            continue
        new_test.append('/' + area + '/' + i)

    print(len(new_test) + len(new_train))
    np.savez(split_name, train=new_train, val=new_val, test=new_test)
