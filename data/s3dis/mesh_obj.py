# -*- coding: utf-8 -*-
import os
from vispy import scene, io
import numpy as np
from collections import namedtuple, defaultdict
from six import iteritems
from os import path
import json
import trimesh


def get_color( i ):
    ''' Parse a 24-bit integer as a RGB color. I.e. Convert to base 256
    Args:
        index: An int. The first 24 bits will be interpreted as a color.
            Negative values will not work properly.
    Returns:
        color: A color s.t. get_index( get_color( i ) ) = i
    '''
    b = ( i ) % 256  # least significant byte
    g = ( i >> 8 ) % 256
    r = ( i >> 16 ) % 256 # most significant byte 
    return r,g,b


label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}

names_to_label = {  '<UNK>':-1,
                               'ceiling':0,
                               'floor':1,
                               'wall':2,
                               'beam':3,
                               'column':4,
                               'window':5,
                               'door':6,
                               'table':7,
                               'chair':8,
                               'sofa':9,
                               'bookcase':10,
                               'board':11,
                               'clutter':12}

def split_rooms(path, fname,color_map):
    """Takes .obj filename and returns dict of object properties for each object in file."""
    obj = {'v': [], 'f': [], 'l': []}
    all_vertices = []
    all_faces = []
    room_vertices = {}
    room_faces = {}
    room_faces_label = {}
    room_faces_color = {}

    with open(fname) as f:
        lines = f.read().splitlines()

    # read Area
    for line in lines:
        if line:
            split_line = line.strip().split(' ', 1)
            if len(split_line) < 2:
                continue

            prefix, value = split_line[0], split_line[1]

            if prefix == 'v':
                coords = [float(value.split(' ')[0]), float(value.split(' ')[1]), float(value.split(' ')[2])]
                all_vertices.append(coords)

            # For files without an 'o' statement
            elif prefix == 'usemtl':
                obj['l'].append(value)
                cls_name = value.split('_')[0]
                label = names_to_label[cls_name]
                room_name = value.split('_')[2] + '_' + value.split('_')[3]
                color = get_color(color_map.index(value))
                if room_name not in room_faces.keys():
                    room_faces[room_name] = []
                    room_faces_label[room_name] = []
                    room_faces_color[room_name] = []

            elif prefix == 'f':
                f = value.split(' ')
                face = []
                for i in f:
                    face.append(int(i.split('/')[0]))
                all_faces.append(face)
                room_faces[room_name].append(face)  # raw face, raw vertices id
                room_faces_label[room_name].append(label)
                room_faces_color[room_name].append(color)

    # print('room_num', len(room_faces.keys()))
    num = 0
    for i in room_faces.keys():
        num += len(room_faces[i])

    print('face_num: ', num)
    print('all vertices:', len(all_vertices))
    print(len(np.unique(all_faces)))

    #####################
    # split Area by rooms
    #####################
    vertices_num = 0
    face_num = 0
    all_vertices = np.array(all_vertices)
    for room_name in list(room_faces.keys()):
        # if room_name == '<UNK>_0':
        #     continue

        # 1. get all the faces of the room, id-1 to fix to 0-indexed.
        # 2. get all the vertices using raw id. 
        ori_verts_ids = np.unique(room_faces[room_name])
        new_verts_ids = ori_verts_ids - 1  # 1-indexed in raw file, should fix to 0-indexed
        room_vertices[room_name] = all_vertices[new_verts_ids]

        # 3. reset face id from 0.
        new_face = []
        for old_face in room_faces[room_name]:
            face = []
            for old_id in old_face:
                new_id = np.argwhere(ori_verts_ids==old_id).flatten()
                face.append(new_id)
            new_face.append(face)
        new_face = np.array(new_face).squeeze(-1)

        labels = np.array(room_faces_label[room_name])
        
        colors = np.array(room_faces_color[room_name])
        mesh = trimesh.Trimesh(vertices=room_vertices[room_name], faces=new_face, process=False)
        if not os.path.exists(path+room_name):
            os.makedirs(path+room_name)
        mesh.export(os.path.join(path + room_name, room_name + '_raw.off'))
        np.savez(os.path.join(path + room_name, room_name + '_colors'), color=colors)

        total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
        centers = (mesh.bounds[1] + mesh.bounds[0]) / 2
        mesh.apply_translation(-centers)
        mesh.apply_scale(1 / total_size)
        if not os.path.exists(path + room_name):
            os.makedirs(path + room_name)
        mesh.export(os.path.join(path + room_name, room_name + '_scaled.off'))
        np.savez(os.path.join(path + room_name, room_name + '_labels'), labels=labels, centers=centers, total_size=total_size)

        face_num+=len(new_face)
        vertices_num += len(room_vertices[room_name])

    print(face_num)
    print(vertices_num)

with open('./assets/semantic_labels.json', 'rb') as f:
    color_map = json.load(f)
for area in ['area_1', 'area_2', 'area_3', 'area_4', 'area_6']:
    path = 'source/' + area + '/3d'
    out_path = 'source/data/' + area + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    split_rooms(out_path, os.path.join(path, 'semantic.obj'),color_map)
