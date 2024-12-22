CUDA_VISIBLE_DEVICES=4 python scripts/compute_mmd_cov_1nn.py \
    --path_to_gt_mesh /s2/yangzhifei/project/MMGDreamer/FRONT/gt_fov90_h8_obj_meshes_refine \
    --path_to_synthesized_mesh /data/yangzhifei/project/MMGDreamer/experiments/xxxx/vis/2049/mmgscene/object_meshes \
    --save_name mmd_cov_1nn
