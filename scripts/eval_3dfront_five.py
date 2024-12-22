from __future__ import print_function

import open3d as o3d # open3d needs to be imported before other packages!
import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from model.SGDiff import SGDiff
from dataset.threedfront_dataset import ThreedFrontDatasetSceneGraph
from helpers.util import bool_flag, preprocess_angle2sincos, batch_torch_destandardize_box_params, descale_box_params, postprocess_sincos2arctan, sample_points
from helpers.metrics_3dfront import validate_constrains, validate_constrains_changes, estimate_angular_std
from helpers.visualize_scene import render_full, render_box
from omegaconf import OmegaConf
import json
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, type=str, default="", help="dataset path")
parser.add_argument('--with_CLIP', type=bool_flag, default=True, help="Load Feats directly instead of points.")
parser.add_argument('--with_image', type=bool_flag, default=False)
parser.add_argument('--manipulate', default=True, type=bool_flag)
parser.add_argument('--exp', default='../experiments/layout_test', help='experiment name')
parser.add_argument('--epoch', type=str, default='100', help='saved epoch')
parser.add_argument('--render_type', type=str, default='txt2shape', help='retrieval, txt2shape, onlybox, mmgscene')
parser.add_argument('--gen_shape', default=False, type=bool_flag, help='infer diffusion')
parser.add_argument('--visualize', default=False, type=bool_flag)
parser.add_argument('--export_3d', default=True, type=bool_flag, help='Export the generated shapes and boxes in json files for future use')
parser.add_argument('--room_type', default='all', help='all, bedroom, livingroom, diningroom, library')
parser.add_argument('--mask_type', default='five', help='one, two, three, four, five, none')
parser.add_argument('--name_render', default='I_R', help='name for render image')


args = parser.parse_args()

room_type = ['all', 'bedroom', 'livingroom', 'diningroom', 'library']


def reseed(num):
    np.random.seed(num)
    torch.manual_seed(num)
    random.seed(num)

def normalize(vertices, scale=1):
    xmin, xmax = np.amin(vertices[:, 0]), np.amax(vertices[:, 0])
    ymin, ymax = np.amin(vertices[:, 1]), np.amax(vertices[:, 1])
    zmin, zmax = np.amin(vertices[:, 2]), np.amax(vertices[:, 2])

    vertices[:, 0] += -xmin - (xmax - xmin) * 0.5
    vertices[:, 1] += -ymin - (ymax - ymin) * 0.5
    vertices[:, 2] += -zmin - (zmax - zmin) * 0.5

    scalars = np.max(vertices, axis=0)
    scale = scale

    vertices = vertices / scalars * scale
    return vertices


def validate_constrains_loop(modelArgs, test_dataset, model, epoch=None, normalized_file=None, cat2objs=None, datasize='large', gen_shape=False):

    test_dataloader_no_changes = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=test_dataset.collate_fn,
        shuffle=False,
        num_workers=0)

    vocab = test_dataset.vocab

    accuracy = {}
    for k in ['left', 'right', 'front', 'behind', 'smaller', 'bigger', 'shorter', 'taller', 'standing on', 'close by', 'symmetrical to', 'total']:
        # compute validation for these relation categories
        accuracy[k] = []

    for i, data in enumerate(test_dataloader_no_changes, 0):
        print(data['scan_id'])

        try:
            dec_objs, dec_triples = data['decoder']['objs'], data['decoder']['tripltes']
            instances = data['instance_id'][0]
            scan = data['scan_id'][0]
        except Exception as e:
            print(e)
            continue

        dec_objs, dec_triples = dec_objs.cuda(), dec_triples.cuda()
        encoded_dec_text_feat, encoded_dec_rel_feat, encoded_dec_image_feat = None, None, None
        if modelArgs['with_CLIP']:
            encoded_dec_text_feat, encoded_dec_rel_feat = data['decoder']['text_feats'].cuda(), data['decoder']['rel_feats'].cuda()
        if modelArgs['with_image']:
            encoded_dec_image_feat = data['decoder']['image_feats'].cuda()
        if not args.with_image:
            encoded_dec_image_feat = torch.zeros_like(encoded_dec_image_feat).cuda()

        all_pred_boxes = []
        all_pred_angles = []

        with torch.no_grad():

            data_dict = model.sample_box_and_shape_mask(dec_objs, dec_triples, encoded_dec_text_feat, encoded_dec_rel_feat, encoded_dec_image_feat, gen_shape=gen_shape, mask_type=args.mask_type)

            boxes_pred, angles_pred = torch.concat((data_dict['sizes'],data_dict['translations']),dim=-1), data_dict['angles']
            shapes_pred = None
            try:
                shapes_pred = data_dict['shapes']
            except:
                print('no shape, only run layout branch.')
            if modelArgs['bin_angle']:
                angles_pred = -180 + (torch.argmax(angles_pred, dim=1, keepdim=True) + 1)* 15.0 # angle (previously minus 1, now add it back)
                boxes_pred_den = batch_torch_destandardize_box_params(boxes_pred, file=normalized_file) # mean, std
            else:
                angles_pred = postprocess_sincos2arctan(angles_pred) / np.pi * 180
                boxes_pred_den = descale_box_params(boxes_pred, file=normalized_file) # min, max


        if args.visualize:
            classes = sorted(list(set(vocab['object_idx_to_name'])))
            # layout and shape visualization through open3d
            print("rendering", [classes[i].strip('\n') for i in dec_objs])
            if model.type_ == 'mmgscene':
                if shapes_pred is not None:
                    shapes_pred = shapes_pred.cpu().detach()
                render_full(data['scan_id'], dec_objs.detach().cpu().numpy(), boxes_pred_den, angles_pred, datasize=datasize,
                classes=classes, render_type=args.render_type, shapes_pred=shapes_pred, store_img=True, render_boxes=False, visual=False, demo=False,epoch=epoch, without_lamp=True, store_path=modelArgs['store_path'])
            else:
                raise NotImplementedError

        all_pred_boxes.append(boxes_pred_den.cpu().detach())
        all_pred_angles.append(angles_pred.cpu().detach())
        # compute constraints accuracy through simple geometric rules
        accuracy = validate_constrains(dec_triples, boxes_pred_den, angles_pred, None, model.vocab, accuracy)

    keys = list(accuracy.keys())
    file_path_for_output = os.path.join(modelArgs['store_path'], f'{test_dataset.eval_type}_accuracy_analysis.txt')
    with open(file_path_for_output, 'w') as file:
        for dic, typ in [(accuracy, "acc")]:
            lr_mean = np.mean([np.mean(dic[keys[0]]), np.mean(dic[keys[1]])])
            fb_mean = np.mean([np.mean(dic[keys[2]]), np.mean(dic[keys[3]])])
            bism_mean = np.mean([np.mean(dic[keys[4]]), np.mean(dic[keys[5]])])
            tash_mean = np.mean([np.mean(dic[keys[6]]), np.mean(dic[keys[7]])])
            stand_mean = np.mean(dic[keys[8]])
            close_mean = np.mean(dic[keys[9]])
            symm_mean = np.mean(dic[keys[10]])
            total_mean = np.mean(dic[keys[11]])
            means_of_mean = np.mean([lr_mean, fb_mean, bism_mean, tash_mean, stand_mean, close_mean, symm_mean])
            print(
                '{} & L/R: {:.2f} & F/B: {:.2f} & Bi/Sm: {:.2f} & Ta/Sh: {:.2f} & Stand: {:.2f} & Close: {:.2f} & Symm: {:.2f}. Total: &{:.2f}'.format(
                    typ, lr_mean,
                    fb_mean, bism_mean, tash_mean, stand_mean, close_mean, symm_mean, total_mean))
            print('means of mean: {:.2f}'.format(means_of_mean))
            file.write(
                '{} & L/R: {:.2f} & F/B: {:.2f} & Bi/Sm: {:.2f} & Ta/Sh: {:.2f} & Stand: {:.2f} & Close: {:.2f} & Symm: {:.2f}. Total: &{:.2f}\n'.format(
                    typ, lr_mean, fb_mean, bism_mean, tash_mean, stand_mean, close_mean, symm_mean, total_mean))
            file.write('means of mean: {:.2f}\n\n'.format(means_of_mean))

def evaluate():
    random.seed(48)
    torch.manual_seed(48)

    argsJson = os.path.join(args.exp, 'args.json')
    assert os.path.exists(argsJson), 'Could not find args.json for experiment {}'.format(args.exp)
    with open(argsJson) as j:
        modelArgs = json.load(j)
    # modelArgs['with_image'] = args.with_image
    normalized_file = os.path.join(args.dataset, 'centered_bounds_{}_trainval.txt').format(modelArgs['room_type'])
    
    test_dataset_no_changes = ThreedFrontDatasetSceneGraph(
        root=args.dataset,
        split='val_scans',
        use_scene_rels=modelArgs['use_scene_rels'],
        use_image_scene_rels=modelArgs['use_image_scene_rels'],
        with_changes=False,
        eval=True,
        eval_type='none',
        with_CLIP=modelArgs['with_CLIP'],
        use_image=modelArgs['with_image'],
        use_SDF=modelArgs['with_SDF'],
        large=modelArgs['large'],
        room_type=args.room_type)

    modeltype_ = modelArgs['network_type']
    modelArgs['store_path'] = os.path.join(args.exp, "vis_" + args.room_type + args.name_render, args.epoch)
    
    replacelatent_ = modelArgs['replace_latent'] if 'replace_latent' in modelArgs else None
    with_changes_ = modelArgs['with_changes'] if 'with_changes' in modelArgs else None
    # args.visualize = False if args.gen_shape == False else args.visualize

    # instantiate the model
    diff_opt = modelArgs['diff_yaml']
    diff_cfg = OmegaConf.load(diff_opt)
    diff_cfg.layout_branch.diffusion_kwargs.train_stats_file = test_dataset_no_changes.box_normalized_stats
    diff_cfg.layout_branch.denoiser_kwargs.using_clip = modelArgs['with_CLIP']  # no
    model = SGDiff(type=modeltype_, diff_opt=diff_cfg, vocab=test_dataset_no_changes.vocab, replace_latent=replacelatent_,
                with_changes=with_changes_, residual=modelArgs['residual'], gconv_pooling=modelArgs['pooling'], clip=modelArgs['with_CLIP'], use_image=modelArgs['with_image'],
                with_angles=modelArgs['with_angles'], separated=modelArgs['separated'])
    model.diff.optimizer_ini()
    model.load_networks(exp=args.exp, epoch=args.epoch, restart_optim=True, load_shape_branch=args.gen_shape)
    if torch.cuda.is_available():
        model = model.cuda()

    model = model.eval()
    cat2objs = None
    
    reseed(47)
    print('\nGeneration Mode')
    validate_constrains_loop(modelArgs, test_dataset_no_changes, model, epoch=args.epoch, normalized_file=normalized_file, cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small', gen_shape=args.gen_shape)

    
if __name__ == "__main__":
    print(torch.__version__)
    evaluate()
