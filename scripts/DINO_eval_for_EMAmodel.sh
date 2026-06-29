python main_teacher.py \
  --dataset_file city\
  --output_dir logs/TEST/R50-MS4-eval \
	-c config/DA/Cityscapes2FoggyCityscapes/DINO_4scale_C2F_self_training.py \
	--eval --resume logs/DINO_SCityscapes2FoggyCityscapes/R50_ms4/best_ema_model.pth \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0