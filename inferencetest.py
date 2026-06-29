import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw

# 请确保这些导入路径在你的项目中是正确的
from main import build_model_main
from util.slconfig import SLConfig
import datasets.transforms as T

# ======================================================
# 路径配置
# ======================================================
MODEL_CONFIG = "config/DA/Cityscapes2FoggyCityscapes/DINO_4scale_C2F.py"
MODEL_CKPT = "logs/DINO_SCityscapes2FoggyCityscapes/checkpoint0003.pth"

IMG_ROOT = r"D:\papercode\DATR-main\data\datasets\cityscapes_foggy\leftImg8bit\val"
ANN_FILE = r"D:\papercode\DATR-main\data\datasets\annotions\cityscapes_val_coco_format_0.02.json"

SAVE_DIR = r"D:\作业\开题\cat2.5\qualitative_results_0.4"

SCORE_THRESH = 0.4
IOU_THRESH = 0.5

# 这里的顺序要和模型训练时的 class map 对应
LABEL_LIST = ['_background_','person', 'car', 'train', 'rider', 'truck', 'mcycle', 'bicycle', 'bus']

# ======================================================
# 颜色定义
# ======================================================
COLORS = {
    "TP": (0, 255, 0),  # Green (正确检测)
    "MISCLS": (0, 0, 255),  # Blue (定位对，类别错)
    "FP": (255, 0, 255),  # Pink (误检)
    "FN": (255, 0, 0),  # Red (漏检)
}


# ======================================================
# IoU 计算
# ======================================================
def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (area1 + area2 - inter)


# ======================================================
# 可视化函数
# ======================================================
def visualize(image, pred_boxes, pred_labels, gt_boxes, gt_labels, save_path):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    gt_used = [False] * len(gt_boxes)
    pred_used = [False] * len(pred_boxes)
    matches = []

    # 1. 匹配过程
    for pi, pbox in enumerate(pred_boxes):
        best_iou, best_gi = 0, -1
        for gi, gbox in enumerate(gt_boxes):
            if gt_used[gi]:
                continue
            iou = compute_iou(pbox, gbox)
            if iou > best_iou:
                best_iou, best_gi = iou, gi

        if best_iou >= IOU_THRESH:
            pred_used[pi] = True
            gt_used[best_gi] = True
            matches.append((pi, best_gi))

    # 2. 绘制 TP / MisCls
    for pi, gi in matches:
        is_correct_cls = (pred_labels[pi] == gt_labels[gi])
        color = COLORS["TP"] if is_correct_cls else COLORS["MISCLS"]
        if pred_labels[pi] != gt_labels[gi]:
            print(
                f"Mismatch! Pred: {pred_labels[pi]} (Name: {LABEL_LIST[pred_labels[pi]] if pred_labels[pi] < len(LABEL_LIST) else '?'}) | GT: {gt_labels[gi]}")

        # 绘制框
        draw.rectangle(pred_boxes[pi].tolist(), outline=color, width=3)

        # 可选：绘制类别文字
        label_text = LABEL_LIST[pred_labels[pi]] if pred_labels[pi] < len(LABEL_LIST) else str(pred_labels[pi])
        # draw.text((pred_boxes[pi][0], pred_boxes[pi][1]), label_text, fill=color)

    # 3. 绘制 FP (多余预测)
    for pi, used in enumerate(pred_used):
        if not used:
            draw.rectangle(pred_boxes[pi].tolist(), outline=COLORS["FP"], width=3)

    # 4. 绘制 FN (漏检 GT)
    for gi, used in enumerate(gt_used):
        if not used:
            draw.rectangle(gt_boxes[gi].tolist(), outline=COLORS["FN"], width=3)

    img.save(save_path)


# ======================================================
# 主程序
# ======================================================
def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # ----------------------
    # 1. 加载模型
    # ----------------------
    args = SLConfig.fromfile(MODEL_CONFIG)
    args.device = 'cuda'
    model, _, postprocessors = build_model_main(args)

    print(f"Loading weights from {MODEL_CKPT} ...")
    ckpt = torch.load(MODEL_CKPT, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval().cuda()

    # ----------------------
    # 2. 定义预处理
    # ----------------------
    # 注意：验证/推理时通常不进行 RandomResize，或者只 Resize 短边
    # 这里保持和训练一致的 Normalization
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ----------------------
    # 3. 加载 GT 标注
    # ----------------------
    print(f"Loading annotations from {ANN_FILE} ...")
    with open(ANN_FILE, 'r') as f:
        coco = json.load(f)

    # 建立文件名到 image_id 的映射
    # 注意：这里处理了反斜杠问题，以匹配不同系统的路径分隔符
    imgid_map = {img['file_name'].replace('\\', '/'): img['id'] for img in coco['images']}

    # 建立 image_id 到 annotations 的映射
    anns_by_imgid = {}
    for ann in coco['annotations']:
        anns_by_imgid.setdefault(ann['image_id'], []).append(ann)

    # ----------------------
    # 4. 收集图片路径
    # ----------------------
    img_rel_paths = []
    for city in os.listdir(IMG_ROOT):
        city_dir = os.path.join(IMG_ROOT, city)
        if not os.path.isdir(city_dir):
            continue
        for name in os.listdir(city_dir):
            if name.endswith('.png') or name.endswith('.jpg'):
                img_rel_paths.append(f"{city}/{name}")

    print(f"Found {len(img_rel_paths)} images. Start inference...")

    # ----------------------
    # 5. 推理循环
    # ----------------------
    for idx, rel_path in enumerate(img_rel_paths):
        # 简单打印进度
        if idx % 10 == 0:
            print(f"Processing [{idx}/{len(img_rel_paths)}]: {rel_path}")

        # --- A. 准备图片 ---
        img_path = os.path.join(IMG_ROOT, rel_path)
        image_ori = Image.open(img_path).convert("RGB")
        w_ori, h_ori = image_ori.size

        # transform 返回 (tensor_img, target)
        image, _ = transform(image_ori, None)

        # --- B. 模型前向 ---
        # 构造 target_sizes: (batch_size, 2)，格式为 [height, width]
        # 这是 DETR/DINO postprocessors 将归一化坐标还原回原图坐标的关键
        target_sizes = torch.tensor([[h_ori, w_ori]], device='cuda')

        with torch.no_grad():
            output = model(image[None].cuda())
            # 直接传入 target_sizes，得到绝对坐标
            results = postprocessors['bbox'](output, target_sizes)[0]

        # --- C. 解析结果 ---
        scores = results['scores'].cpu()
        labels = results['labels'].cpu()
        boxes = results['boxes'].cpu()  # 已经是 xyxy 且是原图尺度

        # 阈值过滤
        # Class 0 is an unused dummy slot because Cityscapes category ids are 1-8.
        keep = (scores > SCORE_THRESH) & (labels > 0)
        pred_boxes = boxes[keep].numpy()
        pred_labels = labels[keep].numpy()

        # --- D. 获取 GT (用于对比) ---
        # 尝试匹配路径找到 image_id
        # 因为 rel_path 是 "city/img.png"，而 JSON 里的 file_name 可能是全路径或相对路径
        # 这里做一个后缀匹配
        img_id = None
        normalized_rel_path = rel_path.replace('\\', '/')
        for k, v in imgid_map.items():
            if k.endswith(normalized_rel_path):
                img_id = v
                break

        if img_id is None:
            # print(f"Warning: No annotation found for {rel_path}")
            continue

        gt_boxes = []
        gt_labels = []
        for ann in anns_by_imgid.get(img_id, []):
            x, y, w, h = ann['bbox']
            gt_boxes.append([x, y, x + w, y + h])  # xywh -> xyxy

            # !!!!!!!! 注意 !!!!!!!!
            # COCO category_id 通常是 1~N，而模型预测通常是 0~N-1
            # 如果发现只有 MisCls (蓝色)，请检查是否需要 -1 或者去掉 -1
            gt_labels.append(ann['category_id'])

        if len(gt_boxes) == 0:
            continue

        # --- E. 可视化并保存 ---
        save_name = os.path.basename(rel_path)
        visualize(
            image_ori,
            np.array(pred_boxes),
            np.array(pred_labels),
            np.array(gt_boxes),
            np.array(gt_labels),
            os.path.join(SAVE_DIR, save_name)
        )

    print("Done!")


if __name__ == '__main__':
    main()
