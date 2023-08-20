import numpy as np
import os

x_list = []
y_list = []

scenes = os.listdir('source/data_color')
train_list=scenes[:1201]
val_list=scenes[1201:1513]
new_train=[]
for scene in train_list:
    new_train.append('/'+scene)

new_val=[]
for scene in val_list:
    new_val.append('/'+scene)

x_list.extend(new_train)
y_list.extend(new_val)

np.savez(f'./split_scannet.npz', train=x_list, val=y_list, test=y_list)