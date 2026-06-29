export CUDA_VISIBLE_DEVICES=0 && python main.py \
--dataset_file city\
--output_dir logs/DINO_SCityscapes2FoggyCityscapes/R50_ms4 -c config/DA/Cityscapes2FoggyCityscapes/DINO_4scale_C2F.py \
--options dn_scalar=100 embed_init_tgt=TRUE \
dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
dn_box_noise_scale=1.0
