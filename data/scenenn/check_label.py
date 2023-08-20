import numpy as np


label_map = np.load('./color_label_map.npz', allow_pickle=True)
face_label_map = label_map['face_label_map'].item()
vertex_label_map = label_map['vertex_label_map'].item()
test_files = ['011', '021', '065', '032', '093', '246', '086', '069', '206',
              '252', '273', '527', '621', '076', '082', '049', '207', '213', '272', '074']

label_to_names = {0: 'unclassified',
                        1: 'wall',
                        2: 'floor',
                        3: 'cabinet',
                        4: 'bed',
                        5: 'chair',
                        6: 'sofa',
                        7: 'table',
                        8: 'door',
                        9: 'window',
                        10: 'bookshelf',
                        11: 'picture',
                        12: 'counter',
                        13: 'blinds',
                        14: 'desk',
                        15: 'shelves',
                        16: 'curtain',
                        18: 'pillow',
                        21: 'clothes',
                        22: 'ceiling',
                        23: 'books',
                        24: 'fridge',
                        25: 'television',
                        26: 'paper',
                        32: 'nightstand',
                        34: 'sink',
                        35: 'lamp',
                        37: 'bag',
                        38: 'strcture',
                        39: 'furniture',
                        40: 'prop'
                        }

print(len(label_to_names.keys()))
print(list(label_to_names.keys()))

train_face_label=  []
train_vertex_label = []
n = 0
for scene_name in face_label_map:
    if scene_name in test_files:
        continue
    a = face_label_map[scene_name][:, 0]
    b = vertex_label_map[scene_name][:, 0]
    train_face_label.extend(np.unique(a))
    train_vertex_label.extend(np.unique(b))
    n +=1
print(n)
train_face_label = np.array(train_face_label)
train_vertex_label = np.array(train_vertex_label)
print(np.unique(train_face_label), len(np.unique(train_face_label)))

for i in np.unique(train_face_label):
    if i not in list(label_to_names.keys()):
        print('miss in table', i)

for i in list(label_to_names.keys()):
    if i not in np.unique(train_face_label):
        print('miss in data', i)


face_label=  []
vertex_label = []
n = 0
for scene_name in face_label_map:
    if scene_name not in test_files:
        continue
    a = face_label_map[scene_name][:, 0]
    b = vertex_label_map[scene_name][:, 0]
    face_label.extend(np.unique(a))
    vertex_label.extend(np.unique(b))
    n +=1
print(n)
face_label = np.array(face_label)
vertex_label = np.array(vertex_label)
print(np.unique(face_label), len(np.unique(face_label)))

for i in np.unique(face_label):
    if i not in list(label_to_names.keys()):
        print('miss in table', i)

for i in list(label_to_names.keys()):
    if i not in np.unique(face_label):
        print('miss in data', i)

for i in np.unique(face_label):
    if i not in np.unique(train_face_label):
        print(i, "not found in train !!!!!!!!!!!!!!!!!")

for i in np.unique(train_face_label):
    if i not in np.unique(face_label):
        print(i, "not found in test!!!!!!!!!!!!!!!!!")
