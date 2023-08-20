import numpy as np
import os

rooms_list = ['rooms_04', 'rooms_05', 'rooms_06', 'rooms_07', 'rooms_08']
x_list = []
y_list = []
z_list = []
for rooms in rooms_list:
    scenes = os.listdir(f'source/{rooms}')
    scenes.sort()
    train_list=scenes[:750]
    # val_list=scenes[750:800]
    test_list=scenes[800:]
    val_list=np.random.choice(test_list,50,replace=False)
    
    new_train=[]
    for scene in train_list:
        new_train.append('/'+rooms+'/'+scene)

    new_val=[]
    for scene in val_list:
        new_val.append('/'+rooms+'/'+scene)
    new_test=[]
    for scene in test_list:
        new_test.append('/'+rooms+'/'+scene)

    x_list.extend(new_train)
    y_list.extend(new_val)
    z_list.extend(new_test)

np.savez(f'./split_syn_1.npz', train=x_list, val=y_list, test=z_list)
