import os
import cv2
from ultralytics import YOLO

# 初始化模型
model = YOLO('runs/detect/train/weights/best.pt')

# 设置路径
val_dir = '/home/public/RT_GENE/rt_gene/rt_gene_standalone/0722_4corner_video/out_picture'
label_dir = os.path.join(val_dir, 'labels')  # 标签目录
os.makedirs(label_dir, exist_ok=True)

# 遍历图像生成标签
for img_name in os.listdir(val_dir):
    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    
    img_path = os.path.join(val_dir, img_name)
    
    # 预测
    results = model(img_path)
    
    # 创建标签文件
    base_name = os.path.splitext(img_name)[0]
    label_path = os.path.join(label_dir, base_name + '.txt')
    
    with open(label_path, 'w') as f:
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls.item())
                conf = box.conf.item()
                x, y, w, h = box.xywhn[0].tolist()  # 归一化坐标
                
                # YOLO格式: class x_center y_center width height
                f.write(f"{cls} {x} {y} {w} {h}\n")
    
    print(f"生成标签: {label_path}")

print(f"所有标签文件已生成在: {label_dir}")