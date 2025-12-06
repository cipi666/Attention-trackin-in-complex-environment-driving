import os
import numpy as np
from sklearn.metrics import roc_auc_score
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import json
from collections import deque
import cv2

# 1. 定义疲劳特征类别
FATIGUE_FEATURES = {
    'closed_eyes': 0,      # 闭眼 - 强疲劳特征
    'open_mouth': 3,       # 张嘴打哈欠 - 强疲劳特征
    'half_closed_eyes': 4  # 半闭眼 - 中等疲劳特征
}

NON_FATIGUE_FEATURES = {
    'open_eyes': 2,        # 睁眼 - 非疲劳
    'closed_mouth': 1      # 闭嘴 - 非疲劳
}

# 2. 时序分析器类
class FatigueAnalyzer:
    def __init__(self, window_size=15):  # 约0.5秒窗口(30fps)
        self.eye_states = deque(maxlen=window_size)  # 存储眼部状态
        self.mouth_states = deque(maxlen=window_size)  # 存储嘴部状态
        self.perclos_threshold = 0.3  # PERCLOS阈值
        self.yawn_threshold = 0.5  # 打哈欠阈值
        
    def update(self, frame_results):
        """更新时序状态"""
        eye_closed = any(cls == FATIGUE_FEATURES['closed_eyes'] for cls in frame_results['classes'])
        eye_half_closed = any(cls == FATIGUE_FEATURES['half_closed_eyes'] for cls in frame_results['classes'])
        mouth_open = any(cls == FATIGUE_FEATURES['open_mouth'] for cls in frame_results['classes'])
        
        # 眼部状态编码: 0=睁眼, 1=半闭眼, 2=闭眼
        eye_state = 2 if eye_closed else (1 if eye_half_closed else 0)
        self.eye_states.append(eye_state)
        
        # 嘴部状态编码: 0=闭嘴, 1=张嘴
        mouth_state = 1 if mouth_open else 0
        self.mouth_states.append(mouth_state)
    
    def calculate_perclos(self):
        """计算闭眼时间比例"""
        if len(self.eye_states) == 0:
            return 0
        closed_count = sum(1 for state in self.eye_states if state == 2)
        return closed_count / len(self.eye_states)
    
    def detect_yawn(self):
        """检测打哈欠模式"""
        if len(self.mouth_states) < 3:
            return 0
        
        # 检测张嘴持续时间
        open_duration = sum(self.mouth_states)
        return open_duration / len(self.mouth_states)
    
    def get_blink_rate(self):
        """计算眨眼频率"""
        if len(self.eye_states) < 3:
            return 0
        
        blink_count = 0
        for i in range(1, len(self.eye_states)-1):
            # 眨眼模式: 睁眼->闭眼->睁眼
            if self.eye_states[i-1] == 0 and self.eye_states[i] == 2 and self.eye_states[i+1] == 0:
                blink_count += 1
        
        # 转换为每分钟眨眼次数 (假设30fps)
        return blink_count / (len(self.eye_states)/30) * 60

# 3. 加载模型
model = YOLO('runs/detect/train/weights/best.pt')

# 4. 创建结果收集器
image_fatigue_probs = []  # 存储每张图像的疲劳概率
image_true_labels = []     # 存储每张图像的真实标签（0=非疲劳, 1=疲劳）

# 5. 调试信息收集器
debug_info = {
    "total_images": 0,
    "images_processed": 0,
    "perclos_values": [],
    "yawn_values": [],
    "blink_rates": []
}

# 6. 设置验证集路径和标签路径
val_dir = '/home/public/RT_GENE/rt_gene/rt_gene_standalone/0722_4corner_video/out_picture'
label_dir = os.path.join(val_dir, 'labels')  # 标签文件目录

# 确保标签目录存在
if not os.path.exists(label_dir):
    os.makedirs(label_dir, exist_ok=True)
    print(f"创建标签目录: {label_dir}")

# 7. 初始化时序分析器
analyzer = FatigueAnalyzer()

# 8. 遍历验证集
for img_name in os.listdir(val_dir):
    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    
    debug_info["total_images"] += 1
    img_path = os.path.join(val_dir, img_name)
    
    # 获取预测结果
    results = model(img_path)
    
    # 获取标签文件路径
    base_name = os.path.splitext(img_name)[0]
    label_path = os.path.join(label_dir, base_name + '.txt')
    
    # 处理预测结果
    frame_results = {'classes': [], 'confidences': []}
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        pred_classes = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        
        frame_results['classes'] = pred_classes.tolist()
        frame_results['confidences'] = confidences.tolist()
    
    # 更新时序分析器
    analyzer.update(frame_results)
    debug_info["images_processed"] += 1
    
    # 计算时序特征
    perclos = analyzer.calculate_perclos()
    yawn_intensity = analyzer.detect_yawn()
    blink_rate = analyzer.get_blink_rate()
    
    debug_info["perclos_values"].append(perclos)
    debug_info["yawn_values"].append(yawn_intensity)
    debug_info["blink_rates"].append(blink_rate)
    
    # 计算综合疲劳概率 (基于生理学模型)
    fatigue_prob = 0.0
    
    # 1. PERCLOS贡献 (40%)
    fatigue_prob += 0.4 * min(perclos / analyzer.perclos_threshold, 1.0)
    
    # 2. 打哈欠贡献 (30%)
    fatigue_prob += 0.3 * min(yawn_intensity / analyzer.yawn_threshold, 1.0)
    
    # 3. 眨眼频率贡献 (20%)
    # 正常眨眼率8-20次/分钟，>20或<8都可能表示疲劳
    if blink_rate < 8 or blink_rate > 20:
        fatigue_prob += 0.2
    
    # 4. 静态特征贡献 (10%)
    if FATIGUE_FEATURES['closed_eyes'] in frame_results['classes']:
        fatigue_prob += 0.1
    elif FATIGUE_FEATURES['open_mouth'] in frame_results['classes']:
        fatigue_prob += 0.05
    
    # 确保概率在[0,1]范围内
    fatigue_prob = min(max(fatigue_prob, 0.0), 1.0)
    image_fatigue_probs.append(fatigue_prob)
    
    # 处理真实标签 (图像级)
    image_fatigue = 0
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    true_cls = int(parts[0])
                    # 只要有一个疲劳特征框，整个图像标记为疲劳
                    if true_cls in FATIGUE_FEATURES.values():
                        image_fatigue = 1
                        break
    
    image_true_labels.append(image_fatigue)

# 9. 调试信息输出
print("\n===== 调试信息 =====")
print(f"总图像数: {debug_info['total_images']}")
print(f"已处理图像数: {debug_info['images_processed']}")
print(f"平均PERCLOS: {np.mean(debug_info['perclos_values']):.4f}")
print(f"平均打哈欠强度: {np.mean(debug_info['yawn_values']):.4f}")
print(f"平均眨眼频率: {np.mean(debug_info['blink_rates']):.2f}次/分钟")

# 10. 确保数据对齐
min_len = min(len(image_fatigue_probs), len(image_true_labels))
if min_len == 0:
    print("\n错误：没有检测到任何有效样本")
    print("调试信息已保存至 debug_info.json")
    with open('debug_info.json', 'w') as f:
        json.dump(debug_info, f, indent=4)
    exit(1)

image_fatigue_probs = image_fatigue_probs[:min_len]
image_true_labels = image_true_labels[:min_len]

# 11. 计算AUC
if len(set(image_true_labels)) < 2:
    print("\n错误：真实标签中只有一种类别")
    print(f"疲劳图像数: {sum(image_true_labels)}")
    print(f"非疲劳图像数: {len(image_true_labels) - sum(image_true_labels)}")
    print("无法计算AUC（需要正负样本都存在）")
else:
    auc_score = roc_auc_score(image_true_labels, image_fatigue_probs)
    print(f"\n图像级疲劳检测AUC: {auc_score:.4f}")
    
    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(image_true_labels, image_fatigue_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (面积 = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率', fontsize=14)
    plt.ylabel('真阳性率', fontsize=14)
    plt.title('疲劳检测ROC曲线', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig('fatigue_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制特征分布图
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(debug_info["perclos_values"], bins=20, color='skyblue', edgecolor='black')
    plt.axvline(analyzer.perclos_threshold, color='red', linestyle='dashed', linewidth=1)
    plt.title('PERCLOS分布', fontsize=14)
    plt.xlabel('PERCLOS值', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    
    plt.subplot(2, 2, 2)
    plt.hist(debug_info["yawn_values"], bins=20, color='lightgreen', edgecolor='black')
    plt.axvline(analyzer.yawn_threshold, color='red', linestyle='dashed', linewidth=1)
    plt.title('打哈欠强度分布', fontsize=14)
    plt.xlabel('打哈欠强度', fontsize=12)
    
    plt.subplot(2, 2, 3)
    plt.hist(debug_info["blink_rates"], bins=20, color='salmon', edgecolor='black')
    plt.axvline(8, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(20, color='red', linestyle='dashed', linewidth=1)
    plt.title('眨眼频率分布', fontsize=14)
    plt.xlabel('眨眼次数/分钟', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    
    plt.subplot(2, 2, 4)
    fatigue_indices = [i for i, label in enumerate(image_true_labels) if label == 1]
    non_fatigue_indices = [i for i, label in enumerate(image_true_labels) if label == 0]
    
    plt.hist([np.array(image_fatigue_probs)[fatigue_indices], 
              np.array(image_fatigue_probs)[non_fatigue_indices]],
             bins=20, color=['red', 'blue'], alpha=0.7, 
             label=['疲劳图像', '非疲劳图像'], edgecolor='black')
    plt.title('疲劳概率分布', fontsize=14)
    plt.xlabel('预测疲劳概率', fontsize=12)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('fatigue_features_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存结果
    with open('fatigue_auc_results.txt', 'w') as f:
        f.write("===== 疲劳检测系统评估报告 =====\n\n")
        f.write(f"评估时间: {np.datetime64('now')}\n")
        f.write(f"总图像数: {min_len}\n")
        f.write(f"疲劳图像比例: {sum(image_true_labels)/len(image_true_labels):.2%}\n")
        f.write(f"疲劳检测AUC: {auc_score:.4f}\n\n")
        
        f.write("===== 关键特征统计 =====\n")
        f.write(f"平均PERCLOS: {np.mean(debug_info['perclos_values']):.4f}\n")
        f.write(f"平均打哈欠强度: {np.mean(debug_info['yawn_values']):.4f}\n")
        f.write(f"平均眨眼频率: {np.mean(debug_info['blink_rates']):.2f}次/分钟\n\n")
        
        f.write("===== 性能评估 =====\n")
        if auc_score > 0.8:
            f.write("性能评级: 优秀 ★★★★★\n")
            f.write("符合挑战杯最高标准 (AUC>0.8)\n")
        elif auc_score > 0.7:
            f.write("性能评级: 良好 ★★★★☆\n")
            f.write("符合挑战杯基础标准 (AUC>0.7)\n")
        else:
            f.write("性能评级: 需改进 ★★★☆☆\n")
            f.write("未达到挑战杯标准 (AUC<0.7)\n")
        
        f.write("\n===== 改进建议 =====\n")
        if auc_score < 0.7:
            f.write("1. 增加时序特征分析窗口大小\n")
            f.write("2. 添加头部姿态分析模块\n")
            f.write("3. 优化模型训练数据平衡\n")
        elif auc_score < 0.8:
            f.write("1. 融合多模态数据（如心率、方向盘操作）\n")
            f.write("2. 使用深度学习时序模型（LSTM/Transformer）\n")
        else:
            f.write("系统性能优异，建议保持当前配置\n")
    
    print(f"结果已保存至 fatigue_auc_results.txt")
    print(f"可视化图表已保存至 fatigue_roc_curve.png 和 fatigue_features_distribution.png")