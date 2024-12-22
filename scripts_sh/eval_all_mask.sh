# mask type three
CUDA_VISIBLE_DEVICES=4 python scripts/eval_3dfront_three.py --exp ./experiments/train_all_image_mask \
    --dataset /s2/yangzhifei/project/MMGDreamer/FRONT \
    --epoch 200 \
    --visualize True \
    --room_type all \
    --render_type mmgscene \
    --gen_shape True \
    --with_image True \
    --mask_type three \
    --name_render I_R

