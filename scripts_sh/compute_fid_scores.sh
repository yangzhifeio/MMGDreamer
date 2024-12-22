# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

CUDA_VISIBLE_DEVICES=3 python scripts/compute_fid_scores_3dfront.py --room bedroom \
    --path_to_real_renderings /s2/yangzhifei/project/MMGDreamer/FRONT/sdf_fov90_h8_wo_lamp_no_stool/small/test \
    --path_to_synthesized_renderings xxxx_render_imgs_path \
    --path_to_test /s2/yangzhifei/project/MMGDreamer/experiments/fid_kid_tmp/

CUDA_VISIBLE_DEVICES=3 python scripts/compute_fid_scores_3dfront.py --room livingroom \
    --path_to_real_renderings /s2/yangzhifei/project/MMGDreamer/FRONT/sdf_fov90_h8_wo_lamp_no_stool/small/test \
    --path_to_synthesized_renderings xxxx_render_imgs_path \
    --path_to_test /s2/yangzhifei/project/MMGDreamer/experiments/fid_kid_tmp/

CUDA_VISIBLE_DEVICES=3 python scripts/compute_fid_scores_3dfront.py --room diningroom \
    --path_to_real_renderings /s2/yangzhifei/project/MMGDreamer/FRONT/sdf_fov90_h8_wo_lamp_no_stool/small/test \
    --path_to_synthesized_renderings xxxx_render_imgs_path \
    --path_to_test /s2/yangzhifei/project/MMGDreamer/experiments/fid_kid_tmp/

CUDA_VISIBLE_DEVICES=3 python scripts/compute_fid_scores_3dfront.py --room all \
    --path_to_real_renderings /s2/yangzhifei/project/MMGDreamer/FRONT/sdf_fov90_h8_wo_lamp_no_stool/small/test \
    --path_to_synthesized_renderings xxxx_render_imgs_path \
    --path_to_test /s2/yangzhifei/project/MMGDreamer/experiments/fid_kid_tmp/

