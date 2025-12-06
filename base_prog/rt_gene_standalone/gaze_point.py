# 视线追踪：通过分析眼睛特征点和视线方向向量，计算用户视线在图像上的落点位置。

import cv2
import csv
import os
import numpy as np
import json

from scipy.interpolate import RBFInterpolator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def load_config(file_path):
    """加载配置并创建全局变量"""
    with open(file_path, 'r') as f:
        config = json.load(f)
    
    # 动态创建全局变量
    globals().update(config)

# 加载配置（自动创建全局变量）
load_config("/home/public/base_prog/rt_gene_standalone/config_set0809.json")

# 变量定义在json中，可忽略以下定义的报错


# 将视线落点的二维物理空间坐标(cm)转换为图像像素坐标(px)
def physical_to_pixel(x, y, # 注视落点的空间坐标x、y
                      U, V  # 数据集视频每帧图片的像素大小（例如1920x1080）
                      ):
    u = SCREEN_WIDE/2 + x
    v = SCREEN_TOP - y

    # 限制范围
    if u <= 0:
        u = 0
    elif u > SCREEN_WIDE:
        u = SCREEN_WIDE
    if v < 0:
        v = 0
    elif v > (SCREEN_TOP - SCREEN_BOTTOM):
        v = SCREEN_TOP - SCREEN_BOTTOM
    
    # 计算【像素大小】和【实际尺寸】的换算关系(px/cm)
    u_conv = U/SCREEN_WIDE
    v_conv = V/(SCREEN_TOP - SCREEN_BOTTOM)

    # cm换算至px
    u = u * u_conv
    v = v * v_conv
    return u, v


def map_gaze(u, v, U, V, cali_path):
    """使用标定结果映射视线点
    
    参数:
        u, v: 原始估计的视线点像素坐标
        U, V: 当前图像尺寸
        cali_path: 标定结果目录路径
        
    返回:
        u_final, v_final: 映射后的视线点像素坐标
    """
    # 加载标定结果
    results_path = cali_path
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"标定文件未找到: {results_path}")
    
    results = np.load(results_path, allow_pickle=True).item()
    
    # 获取关键数据
    mapping_type = results['mapping_type']
    video_info = results['video_info']
    representative_points = results['representative_points']
    theoretical_points = results['theoretical_points']
      
    # 重建映射模型
    if mapping_type == 'tps':
        # 获取有效点
        actual_points = []
        valid_theoretical = []
        for i, rp in enumerate(representative_points):
            if not (np.isnan(rp[0]) or np.isnan(rp[1])):
                actual_points.append(rp)
                valid_theoretical.append(theoretical_points[i])
        
        # 归一化坐标
        width = video_info['width']
        height = video_info['height']
        actual_points_norm = np.array(actual_points) / [width, height]
        theoretical_points_norm = np.array(valid_theoretical)
        
        # 重建TPS模型
        mapping_model = RBFInterpolator(
            actual_points_norm, 
            theoretical_points_norm,
            kernel='thin_plate_spline',
            smoothing=0.1
        )
        
        # 归一化输入点
        point_norm = np.array([u, v]) / [U, V]
        
        # 应用映射
        mapped_norm = mapping_model(point_norm.reshape(1, -1))[0]
        
        # 转换回像素坐标
        u_final = mapped_norm[0] * U
        v_final = mapped_norm[1] * V
    
    elif mapping_type == 'poly':
        # 获取多项式模型参数
        poly = PolynomialFeatures(degree=2)
        model = results['mapping_model']
        
        # 归一化输入点
        point_norm = np.array([u, v]) / [U, V]
        
        # 应用映射
        mapped_norm = model.predict(point_norm.reshape(1, -1))[0]
        
        # 转换回像素坐标
        u_final = mapped_norm[0] * U
        v_final = mapped_norm[1] * V
    
    else:
        raise ValueError(f"不支持的映射类型: {mapping_type}")
    
    # 边界处理    
    if u_final<0 :
        u_final = 0
    elif u_final>U :
        u_final = U
    if v_final<0:
        v_final = 0
    elif v_final>V:
        v_final = V
    
    return u_final, v_final
    
# 根据眼睛特征点、视线向量和相机参数，计算视线在图像上的落点。
def get_gaze_point(image_,          # 数据
                #    base_name,        # 数据名
                   gaze_vector,     # 3D视线方向向量
                   points,          # 4个眼角关键点坐标
                #    path,        # 数据保存路径
                   cali_path,    # 标定结果的路径
                    # 人像画幅左上、右下像素点
                    u1 = FACE_LEFT, 
                    v1 = FACE_TOP, 
                    u2 = FACE_RIGHT, 
                    v2 = FACE_BOTTOM,
                    # 左右眼的物理宽度(cm)
                    wl = EYE_LEFT_WIDTH,
                    wr = EYE_RIGHT_WIDTH,
                    k = K_PX_COMPRESS, # 像素压缩比例k（压缩后/原画）
                    F = FOCUS,       # 相机焦距
                    # 相机在物理空间中的位置偏移
                    ycam = O3_HEIGHT,
                    zcam = O3_TO_SCREEN,
                    mirror= MIRROR         # 数据集的人像是否为镜像
                   ):
    
    # 图像预处理
    if(mirror):
        image = image_
    else:
        image = cv2.flip(image_, 1)     # 水平翻转图像（cv2.flip），确保坐标系一致。# TODO：为什么要翻转啊？疑似没有影响啊？
    U = image.shape[1]              # 获取图像宽高 U, V
    V = image.shape[0]

    # 是否需要翻转（即数据集是否镜像）
    if(mirror):
        u1 = U - u1                     # 翻转眼中心坐标（u1 = U - u1, u2 = U - u2）
        u2 = U - u2

    # 画幅中心坐标（通过对角坐标测算）
    uc = (u1 + u2)*0.5
    vc = (v1 + v2)*0.5
    # 特征点检查：要求points必须是4个点（左右内外眼角）
    if len(points) != 4:
        print("Warning: the number of landmarks is not 4, skip this image")
        return []

    # 眼中心坐标（通过眼角测算）
    if(mirror):
        u0 = U - (points[0][0] + points[1][0] + points[2][0] + points[3][0])/4.0 
    else:
        u0 = (points[0][0] + points[1][0] + points[2][0] + points[3][0])/4.0    
    v0 = (points[0][1] + points[1][1] + points[2][1] + points[3][1])/4.0

    # 像素压缩比例k（压缩后/原画），将像素换算成原画像素
    pl = abs(points[3][0] - points[2][0])/k # 左眼宽度
    pr = abs(points[1][0] - points[0][0])/k # 右眼宽度
    print(f"pl:{pl},pr:{pr}")

    # 面部距离下，每像素对应的物理大小(cm/px)
    # 这里的公式有问题！这里的系数是不需要k的！
    alpha = 0.5 * float(wl/pl + wr/pr)  # TODO:左右镜像？

    # 计算眼中心的z0
    z0 = F * alpha + zcam
    print(f"z0:{z0}")

    # 计算眼中心x0、y0坐标：
    x0 = alpha * (uc - u0)
    y0 = alpha * (vc - v0) + ycam
    print(f"x0:{x0}, y0:{y0}")

    # solve a linear equation
    # 求解视线与屏幕平面（z=0）的交点
    # [x0, y0, z0]->z = 0
    gx, gy, gz = gaze_vector
    if(mirror):
        gx = -gx # 重要，新增

    t = -z0 / gz  # 计算参数t
    x = x0 + t * gx  # 计算x坐标
    y = y0 + t * gy  # 计算y坐标
    print(f"t:{t}")
    print(f"x:{x}, y:{y}")

    #convert to pixel space
    u, v = physical_to_pixel(x, y, U, V)

    # 映射变换：这里需要复用npy文件！
    u_final, v_final = map_gaze(u, v, U, V, cali_path= cali_path)

    # # 写入到csv
    # # 定义CSV表头（固定不变，首次运行时写入）
    # headers = ["image_name",
    #     "gaze_est_x", "gaze_est_y", "gaze_est_z",
    #     "z0", "x0", "y0",
    #     "t", "x", "y",
    #     "u", "v",
    #     "u_final", "v_final"
    # ]

    # # 定义当前运行要写入的数据行（按表头顺序排列）
    # data_row = [base_name, gx, gy, gz, z0, x0, y0, t, x, y, u, v, u_final, v_final]

    # # CSV文件路径
    # csv_path = os.path.join(path, "gaze_intersection_results.csv")

    # # 检查文件是否存在：不存在则创建（含表头），存在则追加数据
    # if not os.path.exists(csv_path):
    #     # 文件不存在时，写入表头+数据
    #     with open(csv_path, "w", newline="", encoding="utf-8") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(headers)  # 写入表头
    #         writer.writerow(data_row) # 写入首行数据
    # else:
    #     # 文件存在时，仅追加数据（不重复表头）
    #     with open(csv_path, "a", newline="", encoding="utf-8") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(data_row)

    return u_final, v_final