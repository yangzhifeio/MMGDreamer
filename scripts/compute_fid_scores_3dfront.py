# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script for computing the FID score between real and synthesized scenes.
"""
import argparse
import os
import sys

import torch

import numpy as np
from PIL import Image

from cleanfid import fid

import shutil

parser = argparse.ArgumentParser(
        description=("Compute the FID scores between the real and the "
                     "synthetic images"))
parser.add_argument(
    "--path_to_real_renderings",
    default="/data/yangzhifei/project/MMGDreamer/FRONT/sdf_fov90_h8_no_stool/small/test",
    help="Path to the folder containing the real renderings"
)
parser.add_argument(
    "--path_to_synthesized_renderings",
    default="/data/yangzhifei/project/MMGDreamer/experiments/train_diningroom/vis/200/render_imgs/mmgscene",
    help="Path to the folder containing the synthesized"
)
# parser.add_argument(
#     "path_to_annotations",
#     help="Path to the folder containing the annotations"
# )
parser.add_argument(
    "--compare_trainval",
    action="store_true",
    help="if compare trainval"
)

parser.add_argument(
    "--room",
    default="bedroom",
    help="if compare trainval, [bedroom, livingroom, diningroom, all]"
)

parser.add_argument(
    "--path_to_test",
    default="/data/yangzhifei/project/MMGDreamer/experiments/fid_kid_tmp/",
    help="path_to_test_real, temp"
)

args = parser.parse_args()

class ThreedFrontRenderDataset(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path = self.dataset[idx].image_path
        img = Image.open(image_path)
        return img


def main():
    
    instruct_scene = False
    room = args.room
    print("testing {}...".format(room))
    room_dict = {'bedroom': ["Bedroom", "MasterBedroom", "SecondBedroom"], 'livingroom': ['LivingDiningRoom','LivingRoom'], 
                 'diningroom': ['LivingDiningRoom','DiningRoom'], 
                 'all': ["Bedroom", "MasterBedroom", "SecondBedroom",'LivingDiningRoom','LivingRoom','DiningRoom']}

    print("Generating temporary a folder with test_real images...")
    path_to_test_real = os.path.join(args.path_to_test, "real")# /tmp/test_real
    if not os.path.exists(path_to_test_real):
        os.makedirs(path_to_test_real)
    real_images = [
        os.path.join(args.path_to_real_renderings, oi)
        for oi in os.listdir(args.path_to_real_renderings)
        if oi.endswith(".png") and oi.split('-')[0] in room_dict[room]
    ]
    for i, fi in enumerate(real_images):
        shutil.copyfile(fi, "{}/{:05d}.png".format(path_to_test_real, i))
    # Number of images to be copied
    N = len(real_images)
    print('number of real images :', len(real_images))

    print("Generating temporary a folder with test_fake images...")
    path_to_test_fake = os.path.join(args.path_to_test, "fake") #/tmp/test_fake/
    if not os.path.exists(path_to_test_fake):
        os.makedirs(path_to_test_fake)

    if not instruct_scene:
        synthesized_images = [
            os.path.join(args.path_to_synthesized_renderings, oi)
            for oi in os.listdir(args.path_to_synthesized_renderings)
            if oi.endswith(".png") and oi.split('-')[0] in room_dict[room]
        ]
    else:
        synthesized_images = [
            os.path.join(args.path_to_synthesized_renderings, oi)
            for oi in os.listdir(args.path_to_synthesized_renderings)
            if oi.endswith(".png") and oi.split('_')[1].split('-')[0] in room_dict[room]
        ]
    print('number of synthesized images :', len(synthesized_images))

    scores = []
    scores2 = []
    file_path_for_output = args.path_to_synthesized_renderings.split("/")[:-2]
    file_path_for_output = os.path.join("/".join(file_path_for_output), args.room + "_fid_kid_result.txt")
    if args.compare_trainval:
        if True:
            for i, fi in enumerate(synthesized_images):
                shutil.copyfile(fi, "{}/{:05d}.png".format(path_to_test_fake, i))

            # Compute the FID score
            fid_score = fid.compute_fid(path_to_test_real, path_to_test_fake, device=torch.device("cuda"))
            print('fid score:', fid_score)
            kid_score = fid.compute_kid(path_to_test_real, path_to_test_fake, device=torch.device("cuda"))
            print('kid score:', kid_score)
            os.system('rm -r %s'%path_to_test_real)
            os.system('rm -r %s'%path_to_test_fake)
            with open(file_path_for_output, 'w') as file:
                file.write('fid score:{}'.format(fid_score))
                file.write('kid score:{}'.format(kid_score))
    else:
        for _ in range(1):
            # np.random.shuffle(synthesized_images)
            # synthesized_images_subset = np.random.choice(synthesized_images, N)
            synthesized_images_subset = synthesized_images
            for i, fi in enumerate(synthesized_images_subset):
                shutil.copyfile(fi, "{}/{:05d}.png".format(path_to_test_fake, i))

            # Compute the FID score
            fid_score = fid.compute_fid(path_to_test_real, path_to_test_fake, device=torch.device("cuda"))

            scores.append(fid_score)
            print('iter: {:d}, fid :{:f}'.format(_, fid_score))
            print('iter: {:d}, fid avg: {:f}'.format(_, sum(scores) / len(scores)) )
            print('iter: {:d}, fid std: {:f}'.format(_, np.std(scores)) )

            fid_score_clip = fid.compute_fid(path_to_test_real, path_to_test_fake, mode="clean", model_name="clip_vit_b_32")
            print('iter: {:d}, fid-clip :{:f}'.format(_, fid_score_clip))

            kid_score = fid.compute_kid(path_to_test_real, path_to_test_fake, device=torch.device("cuda"))
            scores2.append(kid_score)
            print('iter: {:d}, kid: {:f}'.format(_, kid_score) )
            print('iter: {:d}, kid avg: {:f}'.format(_, sum(scores2) / len(scores2)) )
            print('iter: {:d}, kid std: {:f}'.format(_, np.std(scores2)) )
            with open(file_path_for_output, 'w') as file:
                file.write('iter: {:d}, fid :{:f}'.format(_, fid_score))
                file.write('iter: {:d}, fid avg: {:f}'.format(_, sum(scores) / len(scores)))
                file.write('iter: {:d}, fid std: {:f}'.format(_, np.std(scores)))
                file.write('iter: {:d}, fid-clip :{:f}'.format(_, fid_score_clip))
                file.write('iter: {:d}, kid: {:f}'.format(_, kid_score))
                file.write('iter: {:d}, kid avg: {:f}'.format(_, sum(scores2) / len(scores2)))
                file.write('iter: {:d}, kid std: {:f}'.format(_, np.std(scores2)))
                
        os.system('rm -r %s'%path_to_test_real)
        os.system('rm -r %s'%path_to_test_fake)


if __name__ == "__main__":
    main()
