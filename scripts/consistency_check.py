import os
import json

import numpy as np
import torch
import clip
import trimesh
from PIL import Image
import sys
sys.path.append('/s2/yangzhifei/project/MMGDreamer/')
from helpers.util import sample_points

import argparse


import extension.old_chamfer.dist_chamfer as ext
chamfer = ext.chamferDist()
parser = argparse.ArgumentParser(
        description=("test the Chamfer Distance (CD) between two generated objects that are supposed to be identical"))
parser.add_argument(
    "--room",
    default="bedroom",
    help="if compare trainval, [bedroom, livingroom, diningroom, all]"
)
parser.add_argument('--mode', type=str, default='mmgscene', help='retrieval, txt2shape, onlybox, mmgscene')
parser.add_argument(
    "--path_to_test",
    default="/data/yangzhifei/project/MMGDreamer/experiments/train_all/vis/2049",
    help="path_to_test"
)
# parser.add_argument(
#     "path_to_annotations",
#     help="Path to the folder containing the annotations"
# )
parser.add_argument(
    "--catfile",
    default="/data/yangzhifei/project/MMGDreamer/FRONT/classes_all.txt",
    help="catfile"
)
parser.add_argument(
    "--gt_consistency_file",
    default="/data/yangzhifei/project/MMGDreamer/FRONT/classes_all.txt",
    help="gt_consistency_file"
)
args = parser.parse_args()
clip_consis = False
large = False #TODO
no_stool = True #TODO
# bedroom, livingroom, diningroom, all
room_dict = {'bedroom': ["Bedroom", "MasterBedroom", "SecondBedroom"], 'livingroom': ['LivingDiningRoom', 'LivingRoom'],
             'diningroom': ['LivingDiningRoom', 'DiningRoom'],
             'all': ["Bedroom", "MasterBedroom", "SecondBedroom", 'LivingDiningRoom', 'LivingRoom', 'DiningRoom']}
R = room_dict[args.room] 
mode = args.mode 
test_folder = args.path_to_test
print("Testing in {}:".format(R), mode)
catfile = args.catfile      # currently only for class_all
with open(catfile, "r") as f:
    vocab = f.readlines()
if no_stool:
    mapping_full2simple = json.load(open("/data/yangzhifei/project/MMGDreamer/FRONT/mapping.json", "r"))
else:
    mapping_full2simple = json.load(open("/data/yangzhifei/project/MMGDreamer/FRONT/mapping_stool.json", "r"))

vocab = [mapping_full2simple[voc.strip('\n')] + '\n' for voc in vocab] if not large else vocab
classes = dict(zip(sorted(list(set([voc.strip('\n') for voc in vocab]))), range(len(list(set(vocab))))))
classes_r = dict(zip(classes.values(), classes.keys()))

gt_consistency_file = args.gt_consistency_file
assert os.path.exists(gt_consistency_file)
with open(gt_consistency_file, 'r') as f:
    consistency = json.load(f)
if clip_consis:
    clip_model, preprocess = clip.load("ViT-B/32", device='cuda')
    img_folder = os.path.join(test_folder, mode, 'render_object_imgs')
    assert os.path.exists(img_folder)
mesh_folder = os.path.join(test_folder, mode, 'object_meshes')
# mesh_folder = os.path.join(test_folder, 'object_meshes')
chamfer_dist_dict = {}
clip_dist_dict = {}
for scan in consistency['scans']:
    room_name = scan['scan']
    if room_name.split('-')[0] not in R:
        continue
    mesh_folder_ = os.path.join(mesh_folder, room_name)
    if not len(scan['consistency']):
        continue
    for triplet in scan['consistency']:
        sub_id, obj_id = triplet[0], triplet[1]
        sub_name, obj_name = scan["objects"][str(sub_id)], scan["objects"][str(obj_id)]
        if not large:
            sub_name, obj_name = mapping_full2simple[sub_name], mapping_full2simple[obj_name]
        assert sub_name == obj_name
        if sub_name not in chamfer_dist_dict.keys():
            chamfer_dist_dict[sub_name] = []
        if clip_consis:
            img_folder_ = os.path.join(img_folder, room_name)
            if sub_name not in clip_dist_dict.keys():
                clip_dist_dict[sub_name] = []
            sub_img = os.path.join(img_folder_, sub_name +'_'+ str(classes[sub_name]) +'_'+str(sub_id) +'.png')
            obj_img = os.path.join(img_folder_, obj_name + '_' + str(classes[obj_name]) + '_' + str(obj_id) + '.png')
            sub_img = preprocess(Image.open(sub_img)).unsqueeze(0).to('cuda')
            obj_img = preprocess(Image.open(obj_img)).unsqueeze(0).to('cuda')
            with torch.no_grad():
                sub_image_feature = clip_model.encode_image(sub_img)
                obj_image_feature = clip_model.encode_image(obj_img)
                f_distance = torch.norm(sub_image_feature - obj_image_feature)
                clip_dist_dict[sub_name].append(f_distance.cpu().numpy())

        sub_mesh = trimesh.load(os.path.join(mesh_folder_, sub_name + '_' + str(classes[sub_name]) + '_' + str(sub_id) + '.obj'))
        sub_verts = torch.tensor(sub_mesh.vertices, dtype=torch.float32)
        torch.manual_seed(47)
        sub_points = sample_points([sub_verts], 5000)
        sub_points = sub_points[0].unsqueeze(0).float().cuda()
        obj_mesh = trimesh.load(os.path.join(mesh_folder_, obj_name + '_' + str(classes[obj_name]) + '_' + str(obj_id) + '.obj'))
        obj_verts = torch.tensor(obj_mesh.vertices, dtype=torch.float32)
        torch.manual_seed(47)
        obj_points = sample_points([obj_verts], 5000)
        obj_points = obj_points[0].unsqueeze(0).float().cuda()
        dist1, dist2 = chamfer(sub_points,obj_points)
        chamfer_dist = torch.mean(dist1) + torch.mean(dist2)
        chamfer_dist_dict[sub_name].append(chamfer_dist.cpu().numpy())

if clip_consis:
    for cat, dist in clip_dist_dict.items():
        mean_dist = np.mean(np.array(dist))
        print("Category: ", cat, "Pairs: ", len(dist), "CLIP AVG: ", mean_dist)
    num = 0
    sum_f = 0
    for dist_list in clip_dist_dict.values():
        num += len(dist_list)
        sum_f += np.sum(np.array(dist_list))
    print("Total {} pairs, CLIP AVG: ".format(num), sum_f / num)

for cat, dist in chamfer_dist_dict.items():
    mean_dist = np.mean(np.array(dist))
    print("Category: ", cat, "Pairs: ", len(dist), "Chamfer AVG: ", mean_dist)

num = 0
sum_f = 0
for dist_list in chamfer_dist_dict.values():
    num += len(dist_list)
    sum_f += np.sum(np.array(dist_list))
print("Total {} pairs, Chamfer AVG: ".format(num), sum_f / num)






