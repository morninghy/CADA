# import os
# import xml.etree.ElementTree as ET
# from collections import Counter
# from tqdm import tqdm
#
#
# def count_classes_in_xmls(annotations_dir):
#     """
#     遍历指定目录下的所有XML文件，统计其中所有物体的类别及其数量。
#
#     参数:
#     annotations_dir (str): 存放所有XML标注文件的目录。
#     """
#     if not os.path.isdir(annotations_dir):
#         print(f"错误：目录 '{annotations_dir}' 不存在。")
#         return
#
#     class_counter = Counter()
#     xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
#
#     print(f"正在扫描 {len(xml_files)} 个XML文件...")
#
#     for xml_file in tqdm(xml_files):
#         try:
#             tree = ET.parse(os.path.join(annotations_dir, xml_file))
#             root = tree.getroot()
#             for obj in root.findall('object'):
#                 class_name = obj.find('name').text
#                 class_counter[class_name] += 1
#         except ET.ParseError:
#             print(f"警告：无法解析文件 {xml_file}，已跳过。")
#             continue
#
#     print("\n--- 类别统计结果 ---")
#     if not class_counter:
#         print("未找到任何类别标注。")
#         return
#
#     # 打印结果
#     for class_name, count in class_counter.items():
#         print(f"- 类别: {class_name}, 数量: {count}")
#
#     return class_counter
#
#
# # --- 使用示例 ---
# if __name__ == '__main__':
#     # --- !!! 请修改为你的Annotations文件夹路径 !!! ---
#     your_annotations_folder = 'D:\BaiduNetdiskDownload\sim10\Annotations'
#
#     count_classes_in_xmls(your_annotations_folder)
# import json
# import os
# from collections import defaultdict
#
#
# def count_coco_annotations(json_path):
#     """
#     统计 COCO 格式 JSON 文件中每个类别的标注数量。
#
#     参数:
#     json_path (str): COCO JSON 文件的路径。
#     """
#     # 检查文件是否存在
#     if not os.path.exists(json_path):
#         print(f"错误: 文件路径不存在 -> {json_path}")
#         return
#
#     print(f"正在读取文件: {json_path}")
#
#     # --- 步骤 1: 读取 JSON 文件 ---
#     try:
#         with open(json_path, 'r') as f:
#             coco_data = json.load(f)
#     except json.JSONDecodeError:
#         print(f"错误: 文件 '{json_path}' 不是一个有效的 JSON 文件。")
#         return
#     except Exception as e:
#         print(f"读取文件时发生未知错误: {e}")
#         return
#
#     # 检查必要的键是否存在
#     if 'categories' not in coco_data or 'annotations' not in coco_data:
#         print("错误: JSON 文件缺少 'categories' 或 'annotations' 键，请确认是标准的 COCO 格式。")
#         return
#
#     # --- 步骤 2: 创建从 category_id 到 category_name 的映射 ---
#     # 这使得我们可以用人类可读的名称来显示结果
#     category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
#
#     # --- 步骤 3: 初始化计数器 ---
#     # 使用 defaultdict 可以让代码更简洁，当一个键第一次被访问时，会自动创建默认值（这里是 0）
#     category_counts = defaultdict(int)
#
#     # --- 步骤 4: 遍历所有标注并进行计数 ---
#     print("开始统计标注...")
#     annotations = coco_data['annotations']
#     for ann in annotations:
#         category_id = ann.get('category_id')
#         if category_id is not None:
#             # 使用 category_map 找到类别名称
#             if category_id in category_map:
#                 category_name = category_map[category_id]
#                 category_counts[category_name] += 1
#             else:
#                 # 处理数据集中存在但未在 'categories' 列表中定义的 category_id
#                 unknown_category_name = f"未定义ID_{category_id}"
#                 category_counts[unknown_category_name] += 1
#
#     print("统计完成！")
#
#     # --- 步骤 5: 打印结果 ---
#     print("\n" + "=" * 30)
#     print("COCO 数据集类别标注统计结果")
#     print("=" * 30)
#
#     if not category_counts:
#         print("未找到任何标注。")
#         return
#
#     # 为了美观，可以按类别名称排序
#     sorted_categories = sorted(category_counts.keys())
#
#     total_annotations = 0
#     for category_name in sorted_categories:
#         count = category_counts[category_name]
#         print(f"- {category_name:<20}: {count} 个")
#         total_annotations += count
#
#     print("-" * 30)
#     print(f"{'总标注数':<20}: {total_annotations} 个")
#     print("=" * 30 + "\n")
#
#
# if __name__ == '__main__':
#     # --- 请在这里修改您的 COCO JSON 文件路径 ---
#     # 示例:
#     # coco_json_path = 'D:/datasets/coco/annotations/instances_train2017.json'
#     # coco_json_path = '/root/data/my_project/train_coco.json'
#
#     coco_json_path = r'D:\BaiduNetdiskDownload\bdd100k\labels\coco_annotations_bdd100k_train.json'  # <-- 修改这里
#
#     # --- 修改结束 ---
#
#     count_coco_annotations(coco_json_path)