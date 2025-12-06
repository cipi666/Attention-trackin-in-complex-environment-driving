# 视线追踪：通过分析眼睛特征点和视线方向向量，计算用户视线在图像上的落点位置。


# new
from __future__ import print_function, division, absolute_import

import os
import sys
import csv
import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import numpy as np
import json
import argparse

# 添加路径，不需再在命令行新增路径
sys.path.append("/home/public/RT_GENE/rt_gene/rt_gene/src")
# 添加到系统path中,甚至不需要在代码中修改
from rt_gene.extract_landmarks_method_base import LandmarkMethodBase
from rt_gene.gaze_tools import get_phi_theta_from_euler, limit_yaw
from rt_gene.gaze_tools_standalone import euler_from_matrix
from rt_gene.estimate_gaze_pytorch import GazeEstimator

# 参数内置
landmark_estimator = LandmarkMethodBase(
device_id_facedetection="cuda:6",
checkpoint_path_face="../rt_gene/model_nets/SFD/s3fd_facedetector.pth",
checkpoint_path_landmark="../rt_gene/model_nets/phase1_wpdc_vdc.pth.tar",
model_points_file="../rt_gene/model_nets/face_model_68.txt")
script_path = os.path.dirname(os.path.realpath(__file__))
gaze_estimator = GazeEstimator("cuda:6", [os.path.abspath(os.path.join(script_path, '../rt_gene/model_nets/gaze_model_pytorch_vgg16_prl_mpii_allsubjects1.model'))])

def load_config(file_path):
    """加载配置并创建全局变量"""
    with open(file_path, 'r') as f:
        config = json.load(f)
    
    # 动态创建全局变量
    globals().update(config)
    

# 加载配置（自动创建全局变量）
load_config("config_set0809.json")

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
    
# 根据眼睛特征点、视线向量和相机参数，计算视线在图像上的落点。
def get_gaze_point(image_,          # 数据
                   gaze_vector,     # 3D视线方向向量
                   points,          # 4个眼角关键点坐标
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
    print(f"pl:{pl},pr:{pr}")  # 出错了？

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

    return physical_to_pixel(x, y, U, V)

def extract_eye_image_patches(subjects):
    for subject in subjects:
        le_c, re_c, _, _ = subject.get_eye_image_from_landmarks(subject, landmark_estimator.eye_image_size)
        subject.left_eye_color = le_c
        subject.right_eye_color = re_c

def extract_specific_landmarks(subjects):
    specific_points_indices = [36, 39, 42, 45]  # 37、40、43、46号点的索引（索引从0开始）
    specific_points_positions = []
    for subject in subjects:
        landmarks = subject.landmarks
        specific_points = landmarks[specific_points_indices]
        specific_points_positions.append(specific_points)
    return specific_points_positions

def polar_to_cartesian(theta, phi):
    import math
    x = math.cos(theta) * math.sin(phi)
    y = math.sin(theta)
    z = -math.cos(theta) * math.cos(phi)
    return [x, y, z]

def estimate_gaze(color_img):
    global gaze_points, img_size
    
    # 参数内置
    dist_coefficients = np.zeros((1, 5), dtype=np.float32)
    camera_matrix=np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], dtype=np.float32)
    base_name = "None"

    # 记录图像尺寸（用于累计图）
    img_size = (color_img.shape[1], color_img.shape[0])
    
    original_img = color_img.copy()
    img_copy = color_img.copy()



    faceboxes = landmark_estimator.get_face_bb(color_img) #检测人脸边界框
    if len(faceboxes) == 0:
        tqdm.write('Could not find faces in the image')
        return []

    subjects = landmark_estimator.get_subjects_from_faceboxes(color_img, faceboxes)  # 提取人脸区域（subjects）
    extract_eye_image_patches(subjects) # 分割左右眼区域图像

    input_r_list = []
    input_l_list = []
    input_head_list = []
    valid_subject_list = []

    specific_points_positions = extract_specific_landmarks(subjects)

    for idx, subject in enumerate(subjects):
        if subject.left_eye_color is None or subject.right_eye_color is None:
            tqdm.write('Failed to extract eye image patches')
            continue

        success, rotation_vector, _ = cv2.solvePnP(landmark_estimator.model_points,
                                                   subject.landmarks.reshape(len(subject.landmarks), 1, 2),
                                                   cameraMatrix=camera_matrix,
                                                   distCoeffs=dist_coefficients, flags=cv2.SOLVEPNP_DLS)

        if not success:
            tqdm.write('Not able to extract head pose for subject {}'.format(idx))
            continue

        _rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        _rotation_matrix = np.matmul(_rotation_matrix, np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
        _m = np.zeros((4, 4))
        _m[:3, :3] = _rotation_matrix
        _m[3, 3] = 1
        # Go from camera space to ROS space
        _camera_to_ros = [[0.0, 0.0, 1.0, 0.0],
                          [-1.0, 0.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]]
        roll_pitch_yaw = list(euler_from_matrix(np.dot(_camera_to_ros, _m)))
        roll_pitch_yaw = limit_yaw(roll_pitch_yaw)

        phi_head, theta_head = get_phi_theta_from_euler(roll_pitch_yaw)
        

        nose_point = subject.landmarks[30]  # 鼻子尖端关键点
        head_center = (int(nose_point[0]), int(nose_point[1]))
        
        # 计算头部方向终点（简化投影）
        head_dir_length = 50
        head_end_x = int(head_center[0] + head_dir_length * np.sin(phi_head))
        head_end_y = int(head_center[1] - head_dir_length * np.cos(phi_head) * np.sin(theta_head))
        head_end = (head_end_x, head_end_y)

        # 在原图上绘制头部方向
        cv2.arrowedLine(original_img, head_center, head_end, (0, 255, 0), 2, tipLength=0.3)

        # 添加头部姿态文本
        head_text = f"Head: {[round(x, 2) for x in roll_pitch_yaw]}"
        cv2.putText(original_img, head_text, (head_center[0], head_center[1] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 更新：删除图像像素化为224x224
        # face_image_resized = cv2.resize(subject.face_color, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        # head_pose_image = landmark_estimator.visualize_headpose_result(face_image_resized, (phi_head, theta_head))
        head_pose_image = landmark_estimator.visualize_headpose_result(subject.face_color, (phi_head, theta_head))

        # if args.vis_headpose:
        #     plt.axis("off")
        #     plt.imshow(cv2.cvtColor(head_pose_image, cv2.COLOR_BGR2RGB))
        #     plt.show()

        # if args.save_headpose:
        #     # add idx to cope with multiple persons in one image
        #     cv2.imwrite(os.path.join(args.output_path, os.path.splitext(base_name)[0] + '_headpose_%s.jpg'%(idx)), head_pose_image)
        #     # suffix = f'_{frame_count}' if frame_count is not None else f'_{idx}'
        #     # cv2.imwrite(os.path.join(args.output_path, os.path.splitext(base_name)[0] + f'_headpose{suffix}.jpg'), head_pose_image)

        input_r_list.append(gaze_estimator.input_from_image(subject.right_eye_color))
        input_l_list.append(gaze_estimator.input_from_image(subject.left_eye_color))
        input_head_list.append([theta_head, phi_head])
        valid_subject_list.append(idx)

    if len(valid_subject_list) == 0:
        return

    gaze_est_ = gaze_estimator.estimate_gaze_twoeyes(inference_input_left_list=input_l_list,
                                                    inference_input_right_list=input_r_list,
                                                    inference_headpose_list=input_head_list)

    if gaze_est_ is not []:
        gaze_est = gaze_est_.tolist()
        gaze_est = gaze_est[0]
        gaze_est = polar_to_cartesian(gaze_est[0], gaze_est[1])
        print(f"gaze_est:{gaze_est}")

    for subject_id, gaze, headpose in zip(valid_subject_list, gaze_est_.tolist(), input_head_list):
        subject = subjects[subject_id]
        # Build visualizations
        r_gaze_img = gaze_estimator.visualize_eye_result(subject.right_eye_color, gaze)
        l_gaze_img = gaze_estimator.visualize_eye_result(subject.left_eye_color, gaze)
        s_gaze_img = np.concatenate((r_gaze_img, l_gaze_img), axis=1)

        # ===== 视线标注 =====
        # 计算眼睛中心点（简化方法）
        left_eye_center = np.mean(subject.landmarks[36:42], axis=0)  # 左眼关键点
        right_eye_center = np.mean(subject.landmarks[42:48], axis=0)  # 右眼关键点
        
        # 视线方向可视化长度
        gaze_length = 50
        
        # 左眼视线标注
        left_gaze_end = (int(left_eye_center[0] - gaze_length * gaze_est[0]),
                         int(left_eye_center[1] - gaze_length * gaze_est[1]))  # 更新：x、y都取反
        cv2.arrowedLine(original_img, 
                        (int(left_eye_center[0]), int(left_eye_center[1])),
                        left_gaze_end, (255, 0, 0), 2)
        
        # 右眼视线标注
        right_gaze_end = (int(right_eye_center[0] - gaze_length * gaze_est[0]),
                          int(right_eye_center[1] - gaze_length * gaze_est[1]))  # 更新：x、y都取反
        cv2.arrowedLine(original_img, 
                        (int(right_eye_center[0]), int(right_eye_center[1])),
                        right_gaze_end, (0, 0, 255), 2)
        
        # 添加视线向量文本
        gaze_text = f"Gaze: {[round(x, 2) for x in gaze_est]}"
        cv2.putText(original_img, gaze_text, (int(left_eye_center[0]), int(left_eye_center[1]) + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        # 调用gaze_point
        return get_gaze_point(image_=color_img,
                              gaze_vector = gaze_est, 
                              points=specific_points_positions[0],)