import numpy as np
import cv2

# 默认源四边形的四个点（顺序：左上、右上、右下、左下）
# 0801更新：不进行224x224的像素变换后的新标定点
src_pts0 = np.array([
    [-4.412842782, 34.41704282],  # 左上
    [9.207505307, 31.19042582],    # 右上
    [4.697176981, 27.19323878],    # 右下
    [-3.123159061, 30.40557125]     # 左下
], dtype=np.float32)

# 默认目标矩形的四个点（顺序：左上、右上、右下、左下）
dst_pts0 = np.array([
    [-17.75, 25],  # 左上
    [17.75, 25],    # 右上
    [17.75, 5.2],   # 右下
    [-17.75, 5.2]   # 左下
], dtype=np.float32)

def map_point_to_rectangle(x, y, src_pts = src_pts0, dst_pts = dst_pts0):
    """将点 (x, y) 从源四边形映射到目标矩形"""
    point = np.array([x, y], dtype=np.float32).reshape(1, 1, 2)
    # 计算透视变换矩阵
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    transformed_point = cv2.perspectiveTransform(point, H)
    return transformed_point[0][0][0], transformed_point[0][0][1]

# # 示例：映射源四边形的中心点
# center_x = np.mean(src_pts0[:, 0])
# center_y = np.mean(src_pts0[:, 1])
# mapped_x, mapped_y = map_point_to_rectangle(center_x, center_y)
# print(f"映射后的坐标: ({mapped_x}, {mapped_y})")