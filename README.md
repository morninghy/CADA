# CADA

Class-Aware Domain Adaptive Object Detection via Dynamic Cross-Domain Confusion Modeling

本仓库是论文 **CADA: Class-Aware Domain Adaptive Object Detection via Dynamic Cross-Domain Confusion Modeling** 的实现代码。项目面向自动驾驶场景中的无监督域自适应目标检测问题，重点解决真实道路数据中常见的类别不平衡问题：少数类样本少、特征边界弱，容易在跨域适配过程中向外观相近的多数类发生语义漂移。

CADA 基于 DINO 检测器构建，在标准源域监督训练和目标域无标签自训练流程之上，引入动态跨域类别混淆建模、混淆引导的原型对比学习以及逐类细粒度域对齐，使少数类在跨域适配中获得更稳定、更充分的优化信号。

## Highlights

- **CRCM: Class-Relation Construction Module**  
  使用目标域 object query 的预测分布动态维护跨域类别混淆矩阵，显式刻画少数类向多数类漂移的混淆关系。

- **CGCL: Confusion-Guided Contrastive Learning**  
  将混淆矩阵作为 hard negative mining 信号，对容易混淆的类别原型施加更强的分离约束，提升类别原型的判别性。

- **CFA: Class-wise Fine-grained Alignment**  
  为每个类别设置独立的域判别器，在类原型层面分别执行源域和目标域对齐，避免多数类主导整体域对齐过程。

- **Teacher-student self-training**  
  前 36 个 epoch 进行 domain adaptation burn-in，之后进入 EMA teacher 伪标签自训练阶段；目标域 weak augmentation 用于 teacher 生成伪标签，strong augmentation 用于 student 学习。

## Main Results

论文中的主要实验均采用 DINO + ResNet-50 作为基础检测器，并在 Cityscapes、Foggy Cityscapes 和 BDD100K-Daytime 上评估。

### Cityscapes -> Foggy Cityscapes

| Method | Detector | person | rider | car | truck | bus | train | mcycle | bicycle | mAP50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Source-DINO | Transformer | 43.7 | 44.6 | 52.6 | 22.1 | 33.0 | 21.1 | 25.0 | 42.0 | 35.6 |
| DATR | Transformer | 60.6 | 59.2 | 74.9 | 39.5 | 62.1 | 27.5 | 45.5 | 53.5 | 52.8 |
| **CADA** | Transformer | **60.7** | **63.2** | **75.3** | 32.2 | 58.5 | **35.9** | **46.9** | **59.3** | **54.5** |

### Cityscapes -> BDD100K-Daytime

| Method | Detector | person | rider | car | truck | bus | train | bicycle | mAP50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DATR | Transformer | 57.7 | 37.7 | 75.8 | 31.3 | **35.3** | 28.8 | 26.8 | 41.9 |
| **CADA** | Transformer | 56.4 | **39.2** | **76.1** | **32.5** | 33.2 | **29.4** | **31.8** | **42.6** |

### Ablation on Cityscapes -> Foggy Cityscapes

| Backbone alignment | CRCM | CGCL | CFA | mAP50 |
| --- | --- | --- | --- | ---: |
| - | - | - | - | 35.6 |
| yes | - | - | - | 42.5 |
| yes | - | yes | - | 48.7 |
| yes | yes | yes | - | 53.7 |
| yes | yes | yes | yes | **54.5** |

## Repository Structure

```text
CADA-main/
|-- main.py                         # standard DA training and evaluation entry
|-- main_teacher.py                 # EMA teacher-student self-training entry
|-- engine.py                       # train/eval loops, pseudo-label training logic
|-- inference.py                    # image-folder inference and visualization example
|-- inference_ema_model.py          # inference with checkpoint['ema_model']
|-- config/
|   |-- DINO/                       # base DINO configs
|   `-- DA/
|       |-- Cityscapes2FoggyCityscapes/
|       |-- Cityscapes2BDD100k/
|       `-- Sim10k2Cityscapes/
|-- datasets/
|   |-- DAcoco.py                   # domain-adaptive COCO-format datasets
|   `-- da_transforms.py            # DA data augmentation
|-- models/dino/
|   |-- dino.py                     # DINO + CADA modules and losses
|   |-- DA_utils.py                 # domain adaptation utilities and prototypes
|   |-- EMA.py                      # EMA teacher / cosine EMA helpers
|   `-- ops/                        # MultiScaleDeformableAttention CUDA op
`-- scripts/
    |-- DINO_train.sh
    |-- DINO_train_self_training.sh
    |-- DINO_eval.sh
    `-- DINO_eval_for_EMAmodel.sh
```

## Environment

The paper experiments were run with:

- Windows 11
- Python 3.9
- PyTorch 2.0.1
- CUDA 11.8
- Single NVIDIA L20 GPU with 48 GB memory

Install dependencies:

```bash
conda create -n cada python=3.9 -y
conda activate cada

pip install -r requirements.txt
```

Build the deformable attention operator:

```bash
cd models/dino/ops
python setup.py build install
cd ../../..
```

If the operator build fails, check that your local CUDA, PyTorch and compiler versions are compatible.

## Data Preparation

The project uses COCO-format annotations for domain-adaptive detection. The supported dataset switches are:

| `--dataset_file` | Source -> Target | Config directory |
| --- | --- | --- |
| `city` | Cityscapes -> Foggy Cityscapes | `config/DA/Cityscapes2FoggyCityscapes/` |
| `city2bdd100k` | Cityscapes -> BDD100K-Daytime | `config/DA/Cityscapes2BDD100k/` |
| `sim2city` | Sim10K -> Cityscapes | `config/DA/Sim10k2Cityscapes/` |

Before training, edit the dataset paths in `datasets/DAcoco.py`.

For Cityscapes -> Foggy Cityscapes, update `PATHS_Source` and `PATHS_Target` inside `build_city_DA`:

```python
PATHS_Source = {
    "train": ("path/to/cityscapes/leftImg8bit/train",
              "path/to/annotations/cityscapes_train_coco_format.json"),
    "val": ("path/to/cityscapes/leftImg8bit/val",
            "path/to/annotations/cityscapes_val_coco_format.json"),
}

PATHS_Target = {
    "train": ("path/to/cityscapes_foggy/leftImg8bit/train",
              "path/to/annotations/cityscapes_train_coco_format_0.02.json"),
    "val": ("path/to/cityscapes_foggy/leftImg8bit/val",
            "path/to/annotations/cityscapes_val_coco_format_0.02.json"),
}
```

For Cityscapes-style experiments, class id `0` is reserved as an unused dummy slot, so configs use `num_classes = 9` for the eight foreground classes:

```text
person, car, train, rider, truck, motorcycle, bicycle, bus
```

## Training

### 1. Burn-in domain adaptation

```bash
bash scripts/DINO_train.sh
```

Equivalent command:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset_file city \
  --output_dir logs/DINO_SCityscapes2FoggyCityscapes/R50_ms4 \
  -c config/DA/Cityscapes2FoggyCityscapes/DINO_4scale_C2F.py \
  --options dn_scalar=100 embed_init_tgt=TRUE \
  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
  dn_box_noise_scale=1.0
```

### 2. Teacher-student self-training

```bash
bash scripts/DINO_train_self_training.sh
```

Equivalent command:

```bash
CUDA_VISIBLE_DEVICES=0 python main_teacher.py \
  --dataset_file city \
  --output_dir logs/DINO_SCityscapes2FoggyCityscapes/R50_ms4 \
  -c config/DA/Cityscapes2FoggyCityscapes/DINO_4scale_C2F_self_training.py \
  --options dn_scalar=100 embed_init_tgt=TRUE \
  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
  dn_box_noise_scale=1.0
```

The self-training config uses:

- `epochs = 46`
- `burn_epochs = 36`
- `cm_warmup_epochs = 10`
- `pseudo_label_threshold = 0.3`
- `ema_decay_teacher = 0.9997`
- `da_proto_loss_coef = 0.1`
- `da_global_proto_coef = 0.1`

During training, the code saves useful checkpoints and logs under `--output_dir`, including:

- `checkpoint1.5.pth`: latest checkpoint
- `best_ema_teacher.pth`: best EMA teacher during burn-in
- `best_ema_model.pth`: best final EMA model during self-training
- `log.txt`, `log_best.txt`: training and best-metric logs

## Evaluation

Evaluate a regular checkpoint:

```bash
python main.py \
  --dataset_file city \
  --output_dir logs/TEST/R50-MS4-eval \
  -c config/DA/Cityscapes2FoggyCityscapes/DINO_4scale_C2F.py \
  --eval \
  --resume weights/city2fog_city/best_model.pth \
  --options dn_scalar=100 embed_init_tgt=TRUE \
  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
  dn_box_noise_scale=1.0
```

Evaluate the EMA model saved as `checkpoint['ema_model']`:

```bash
python main_teacher.py \
  --dataset_file city \
  --output_dir logs/TEST/R50-MS4-eval \
  -c config/DA/Cityscapes2FoggyCityscapes/DINO_4scale_C2F_self_training.py \
  --eval \
  --resume logs/DINO_SCityscapes2FoggyCityscapes/R50_ms4/best_ema_model.pth \
  --options dn_scalar=100 embed_init_tgt=TRUE \
  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
  dn_box_noise_scale=1.0
```

Evaluation uses COCO-style metrics and prints AP for each class.

## Inference and Visualization

For standard checkpoints with `checkpoint['model']`, edit the paths in `inference.py`:

```python
model_config_path = "config/DA/Cityscapes2FoggyCityscapes/DINO_4scale_C2F.py"
model_checkpoint_path = "weights/city2fog_city/best_model.pth"
img_dir = "path/to/images"
save_dir = "path/to/save/visualizations"
```

Then run:

```bash
python inference.py
```

For EMA checkpoints with `checkpoint['ema_model']`, use `inference_ema_model.py` and update:

```python
model_config_path = "path/to/config.py"
model_checkpoint_path = "path/to/ema_model.pth"
img_dir = "path/to/images"
save_dir = "path/to/save/visualizations"
```

Then run:

```bash
python inference_ema_model.py
```

## Notes

- `datasets/DAcoco.py` currently contains local absolute paths. Replace them before running experiments.
- `num_classes` follows the maximum category id convention used by this codebase. For Cityscapes-style labels, use `9` because class id `0` is unused and foreground ids are `1` to `8`.
- `CFA` discriminators are used during training only. They do not add inference cost.
- The repository contains a prebuilt Windows deformable attention artifact under `models/dino/ops/`, but rebuilding the op in your own environment is recommended.

## Citation

If this project is useful for your research, please cite the paper:

```bibtex
@article{mu2026cada,
  title  = {CADA: Class-Aware Domain Adaptive Object Detection via Dynamic Cross-Domain Confusion Modeling},
  author = {Mu, Huiyu and Hu, Yun and Zhou, Liming and Yang, Xiangyang and Zhang, Wanjun},
  journal = {Preprint},
  year   = {2026}
}
```

## Acknowledgement

This codebase is built on top of DINO-style transformer detection components and extends them for class-aware unsupervised domain adaptive object detection.
