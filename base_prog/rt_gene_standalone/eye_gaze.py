
'''可以通过命令行参数设置输入图像的路径
    给定一组图像，依次为每张图像返回gaze_result，
    用[x, y, z]表示右手直角坐标系里的向量坐标
    这个向量应该代表两眼中心到注视点的向量
    u, v = get_gaze_point(image_=image, gaze_vector=gaze_est, u1=0, v1=0, u2=0, v2=0, points=specific_points_positions[0], wl=0, wr=0, F=0, ycam = 38.95, zcam=-8)
    参数待测量'''


from __future__ import print_function, division, absolute_import

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from rt_gene.extract_landmarks_method_base import LandmarkMethodBase
from rt_gene.gaze_tools import get_phi_theta_from_euler, limit_yaw
from rt_gene.gaze_tools_standalone import euler_from_matrix
# import sys
# sys.path.append()
from gaze_point import get_gaze_point
from gaze_object import connect_object

script_path = os.path.dirname(os.path.realpath(__file__))


def load_camera_calibration(calibration_file):
    import yaml
    with open(calibration_file, 'r') as f:
        cal = yaml.safe_load(f)

    dist_coefficients = np.array(cal['distortion_coefficients']['data'], dtype='float32').reshape(1, 5)
    camera_matrix = np.array(cal['camera_matrix']['data'], dtype='float32').reshape(3, 3)

    return dist_coefficients, camera_matrix

def extract_specific_landmarks(subjects):
    specific_points_indices = [36, 39, 42, 45]  # 37、40、43、46号点的索引（索引从0开始）
    specific_points_positions = []
    for subject in subjects:
        landmarks = subject.landmarks
        specific_points = landmarks[specific_points_indices]
        specific_points_positions.append(specific_points)
    return specific_points_positions

def extract_eye_image_patches(subjects):
    for subject in subjects:
        le_c, re_c, _, _ = subject.get_eye_image_from_landmarks(subject, landmark_estimator.eye_image_size)
        subject.left_eye_color = le_c
        subject.right_eye_color = re_c


def estimate_gaze(base_name, color_img, dist_coefficients, camera_matrix, visual_output_dir):
    original_img = color_img.copy()
    img_copy = color_img.copy()

    faceboxes = landmark_estimator.get_face_bb(color_img)
    if len(faceboxes) == 0:
        tqdm.write('Could not find faces in the image')
        return []

    subjects = landmark_estimator.get_subjects_from_faceboxes(color_img, faceboxes)
    extract_eye_image_patches(subjects)

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


        face_image_resized = cv2.resize(subject.face_color, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        head_pose_image = landmark_estimator.visualize_headpose_result(face_image_resized, (phi_head, theta_head))

        if args.vis_headpose:
            plt.axis("off")
            plt.imshow(cv2.cvtColor(head_pose_image, cv2.COLOR_BGR2RGB))
            plt.show()

        if args.save_headpose:
            # add idx to cope with multiple persons in one image
            cv2.imwrite(os.path.join(args.output_path, os.path.splitext(base_name)[0] + '_headpose_%s.jpg'%(idx)), head_pose_image)

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
        
        # 视线方向长度
        gaze_length = 30
        
        # 左眼视线标注
        left_gaze_end = (int(left_eye_center[0] + gaze_length * gaze_est[0]),
                         int(left_eye_center[1] - gaze_length * gaze_est[1]))  # y取反
        cv2.arrowedLine(original_img, 
                        (int(left_eye_center[0]), int(left_eye_center[1])),
                        left_gaze_end, (255, 0, 0), 2)
        
        # 右眼视线标注
        right_gaze_end = (int(right_eye_center[0] + gaze_length * gaze_est[0]),
                          int(right_eye_center[1] - gaze_length * gaze_est[1]))  # y取反
        cv2.arrowedLine(original_img, 
                        (int(right_eye_center[0]), int(right_eye_center[1])),
                        right_gaze_end, (0, 0, 255), 2)
        
        # 添加视线向量文本
        gaze_text = f"Gaze: {[round(x, 2) for x in gaze_est]}"
        cv2.putText(original_img, gaze_text, (int(left_eye_center[0]), int(left_eye_center[1]) + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        #apply gaze_point
        u, v = get_gaze_point(image_=image, gaze_vector=gaze_est, u1=0, v1=0, u2=656, v2=369, points=specific_points_positions[0], wl=3.94, wr=3.89, F=(22.9 * 1080 /3.5 * 3)/4, ycam = 35.5, zcam=-16)
        print(f"u:{u}, v:{v}")

        #暂时注释
        # base_name_ = os.path.splitext(image_file_name)[0]
        # # 将"frame"替换为"coord"生成坐标文件名
        # coord_file_name = base_name_.replace("frame", "coord") + ".txt"
        # coord_file_path = os.path.join("/home/public/RT_GENE/rt_gene/rt_gene_standalone/coord", coord_file_name)
        
        #target_list = connect_object(u, v, filename=coord_file_path)

        cv2.circle(original_img, (int(u), int(v)), radius=20, color=(0, 0, 255), thickness=-1)
        #print(target_list)
        
        visual_output_path = os.path.join(visual_output_dir, base_name)
        cv2.imwrite(visual_output_path, original_img)

        if args.vis_gaze:
            plt.axis("off")
            plt.imshow(cv2.cvtColor(s_gaze_img, cv2.COLOR_BGR2RGB))
            plt.show()

def add_parser():
    parser = argparse.ArgumentParser(description='Estimate gaze from images')
    parser.add_argument('im_path', type=str, default=os.path.abspath(os.path.join(script_path, './test_gaze/')),
                        nargs='?', help='Path to an image or a directory containing images')
    parser.add_argument('--calib-file', type=str, dest='calib_file', default=None, help='Camera calibration file')
    parser.add_argument('--vis-headpose', dest='vis_headpose', action='store_true', help='Display the head pose images')
    parser.add_argument('--no-vis-headpose', dest='vis_headpose', action='store_false', help='Do not display the head pose images')
    parser.add_argument('--save-headpose', dest='save_headpose', action='store_true', help='Save the head pose images')
    parser.add_argument('--no-save-headpose', dest='save_headpose', action='store_false', help='Do not save the head pose images')
    parser.add_argument('--vis-gaze', dest='vis_gaze', action='store_true', help='Display the gaze images')
    parser.add_argument('--no-vis-gaze', dest='vis_gaze', action='store_false', help='Do not display the gaze images')
    parser.add_argument('--save-gaze', dest='save_gaze', action='store_true', help='Save the gaze images')
    parser.add_argument('--save-estimate', dest='save_estimate', action='store_true', help='Save the predictions in a text file')
    parser.add_argument('--no-save-gaze', dest='save_gaze', action='store_false', help='Do not save the gaze images')
    parser.add_argument('--output_path', type=str, default=os.path.abspath(os.path.join(script_path, './samples_gaze/out')),
                        help='Output directory for head pose and gaze images')
    parser.add_argument('--models', nargs='+', type=str, default=[os.path.abspath(os.path.join(script_path, '../rt_gene/model_nets/gaze_model_pytorch_vgg16_prl_mpii_allsubjects1.model'))],
                        help='List of gaze estimators')
    parser.add_argument('--device-id-pytorch', dest="device_id_pytorch", type=str, default='cuda:6', help='Pytorch device id. Set to "cpu:0" to disable cuda')
    

    parser.set_defaults(vis_gaze=False)
    parser.set_defaults(save_gaze=False)
    parser.set_defaults(vis_headpose=False)
    parser.set_defaults(save_headpose=False)
    parser.set_defaults(save_estimate=False)

    args = parser.parse_args()
    return args

import math

def polar_to_cartesian(theta, phi):

    x = math.cos(theta) * math.sin(phi)
    y = math.sin(theta)
    z = -math.cos(theta) * math.cos(phi)
    return [x, y, z]


if __name__ == '__main__':
    
    args = add_parser()
    image_path_list = []
    if os.path.isfile(args.im_path):
        image_path_list.append(os.path.split(args.im_path)[1])
        args.im_path = os.path.split(args.im_path)[0]
    elif os.path.isdir(args.im_path):
        for image_file_name in sorted(os.listdir(args.im_path)):
            if image_file_name.lower().endswith('.jpg') or image_file_name.lower().endswith('.png') or image_file_name.lower().endswith('.jpeg'):
                if '_gaze' not in image_file_name and '_headpose' not in image_file_name:
                    image_path_list.append(image_file_name)
    else:
        tqdm.write('Provide either a path to an image or a path to a directory containing images')
        sys.exit(1)

    tqdm.write('Loading networks')
    landmark_estimator = LandmarkMethodBase(device_id_facedetection=args.device_id_pytorch,
                                            checkpoint_path_face=os.path.abspath(os.path.join(script_path, "../rt_gene/model_nets/SFD/s3fd_facedetector.pth")),
                                            checkpoint_path_landmark=os.path.abspath(
                                                os.path.join(script_path, "../rt_gene/model_nets/phase1_wpdc_vdc.pth.tar")),
                                            model_points_file=os.path.abspath(os.path.join(script_path, "../rt_gene/model_nets/face_model_68.txt")))

    from rt_gene.estimate_gaze_pytorch import GazeEstimator

    gaze_estimator = GazeEstimator(args.device_id_pytorch, args.models)
    
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    # 可视化输出文件夹
    visual_output_dir = os.path.join(args.im_path, "out_visual")
    os.makedirs(visual_output_dir, exist_ok=True)
    for image_file_name in tqdm(image_path_list):
        tqdm.write('Estimate gaze on ' + image_file_name)
        image = cv2.imread(os.path.join(args.im_path, image_file_name))
        if image is None:
            tqdm.write('Could not load ' + image_file_name + ', skipping this image.')
            continue

        if args.calib_file is not None:
            _dist_coefficients, _camera_matrix = load_camera_calibration(args.calib_file)
        
        else:
            im_width, im_height = image.shape[1], image.shape[0]
            tqdm.write('WARNING!!! You should provide the camera calibration file, otherwise you might get bad results. Using a crude approximation!')
            _dist_coefficients, _camera_matrix = np.zeros((1, 5)), np.array(
                [[im_height, 0.0, im_width / 2.0], [0.0, im_height, im_height / 2.0], [0.0, 0.0, 1.0]])

        print(image_file_name)
        estimate_gaze(image_file_name, image, _dist_coefficients, _camera_matrix, visual_output_dir)
        
        
        

  