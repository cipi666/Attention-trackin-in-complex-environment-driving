"""
代码功能:
    输入:
    - 录屏的标定视频文件(要求:前500s内有"开始录制"的闪光出现)
    - 对应为: VideoCalibrationAnalyzer的video_path
    输出:
    - 路径暂时设为相对路径下文件夹"calibration_result",共有5个文件:
    - calibration_result.npy:   映射关系(最重要的文件),
                                **如需运行eye_gaze.py,需要手动复制到待处理的video文件夹下**, 也可以重设文件输出路径~
    - mapping_error.png:        显示各标定点的映射误差和误差分布
    - points_comparison.png:    展示注视代表点(星星)与理论点(红点)的对比关系,直观地分析映射关系是否正常
    - raw_gaze_points.png:      散点图, 显示所有标定段内提取的原始眼动点位置
    - time_based_segments.png:  显示基于时间的标定段划分，展示每个标定点的开始帧和结束帧
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import RBFInterpolator
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import os
import logging
import time

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoCalibrationAnalyzer:
    def __init__(self, video_path, theoretical_points):
        """
        初始化视频标定分析器
        
        参数：
            video_path: 录屏视频文件路径
            theoretical_points: 标定点的理论坐标列表 [(x1, y1), (x2, y2), ...]
        """
        self.video_path = video_path
        self.theoretical_points = theoretical_points
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 计算每帧的持续时间(毫秒)
        self.frame_duration_ms = 1000.0 / self.fps
        
        # 将理论点转换为实际像素坐标
        self.actual_theoretical_points = [
            (int(x * self.width), int(y * self.height)) 
            for x, y in theoretical_points
        ]
        
        # 存储处理结果
        self.calibration_segments = []
        self.gaze_points = []
        self.representative_points = []
        self.mapping_model = None
        self.mapping_type = None
        
        logger.info(f"视频信息: {self.width}x{self.height}, {self.fps:.2f} FPS, {self.total_frames} 帧")
        logger.info(f"理论点数量: {len(self.actual_theoretical_points)}")
        logger.info(f"每帧持续时间: {self.frame_duration_ms:.2f} 毫秒")
    
    def calculate_time_based_segments(self):
        """
        根据时间点计算标定段，检测第一个标定点前的150ms大面积闪烁
        时间点定义:
          第0ms: 检测到的大面积闪烁开始时间
          第150ms: 大面积闪烁结束，第一个标定点开始闪烁
          第300ms: 第一个标定点闪烁结束，标定点高亮激活
          第3150ms: 标定点结束，再次闪烁
          第3300ms: 显示3秒缓冲倒计时(非最后一点)
          第6300ms: 进入下一个标定点
        """
        logger.info("开始基于时间点计算标定段...")
        self.calibration_segments = []
        
        # 1. 检测大面积闪烁开始帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        large_flash_start_frame = None
        
        logger.info("检测大面积闪烁开始帧...")
        for frame_idx in tqdm(range(0, min(30000, self.total_frames))):  # 检查前30000帧,60fps下为500s,足够长.
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 检测大面积闪烁
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)  # 高亮度区域
            bright_area_ratio = np.sum(binary == 255) / (self.width * self.height)
            
            if bright_area_ratio > 0.75:  # 超过75%区域高亮
                large_flash_start_frame = frame_idx
                logger.info(f"检测到大面积闪烁: 帧 {frame_idx}, 高亮区域占比 {bright_area_ratio:.2f}")
                break
        
        if large_flash_start_frame is None:
            logger.warning("未检测到大面积闪烁，使用默认时间基准 (0ms开始)")
            large_flash_start_frame = 0
        else:
            logger.info(f"大面积闪烁开始帧: {large_flash_start_frame}")
        
        # 计算大面积闪烁开始时间(毫秒)
        large_flash_start_ms = large_flash_start_frame * self.frame_duration_ms
        
        # 2. 计算标定段
        # 每个标定点的总时长(毫秒)
        point_duration = 6300
        
        # 遍历所有标定点
        for point_idx in range(len(self.theoretical_points)):
            # 计算当前标定点的开始时间(毫秒)
            start_time_ms = large_flash_start_ms + point_idx * point_duration
            
            # 激活时间段偏移量（相对于标定点开始时间）
            if point_idx == 0:
                # 第一个标定点：跳过150ms大面积闪烁 + 150ms标定点闪烁 + 舍弃前1s
                activate_start_offset = 150 + 150 + 1000  # 300ms + 1000ms = 1300ms
                activate_end_offset = 150 + 150 + 3000    # 300ms + 3000ms = 3300ms
            else:
                # 后续标定点：直接150ms标定点闪烁 + 舍弃前1s
                activate_start_offset = 150 + 1000  # 1150ms
                activate_end_offset = 150 + 3000    # 3150ms
            
            # 计算激活段的开始和结束时间(毫秒)
            activate_start_ms = start_time_ms + activate_start_offset
            activate_end_ms = start_time_ms + activate_end_offset
            
            # 转换为帧号
            start_frame = int(round(activate_start_ms / self.frame_duration_ms))
            end_frame = int(round(activate_end_ms / self.frame_duration_ms))
            
            # 确保帧号在有效范围内
            if start_frame < 0:
                start_frame = 0
            if end_frame >= self.total_frames:
                end_frame = self.total_frames - 1
            
            # 如果时间段有效，则添加
            if end_frame > start_frame:
                self.calibration_segments.append((point_idx, start_frame, end_frame))
                logger.info(f"标定点 {point_idx}: 激活时间 {activate_start_ms:.1f}-{activate_end_ms:.1f}ms, "
                           f"对应帧 {start_frame}-{end_frame} (时长: {end_frame - start_frame}帧)")
            else:
                logger.warning(f"标定点 {point_idx} 的时间段无效: 开始帧={start_frame}, 结束帧={end_frame}")
        
        logger.info(f"成功计算 {len(self.calibration_segments)} 个标定段")
        return self.calibration_segments
    
    def extract_gaze_points(self, gaze_detector, sample_rate=1):
        """
        提取每个标定段内的视线落点
        
        参数：
            gaze_detector: 视线落点检测函数，接收帧图像返回 (x, y)
            sample_rate: 采样率，每几帧采样一次
        """
        if not self.calibration_segments:
            raise ValueError("请先运行 calculate_time_based_segments() 方法")
        
        logger.info("开始提取视线落点...")
        self.gaze_points = [[] for _ in range(len(self.actual_theoretical_points))]
        
        total_frames = sum([end - start + 1 for _, start, end in self.calibration_segments])
        processed_frames = 0
        pbar = tqdm(total=total_frames)
        
        for seg in self.calibration_segments:
            point_idx, start, end = seg
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            
            for frame_idx in range(start, end + 1):
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_rate == 0:
                    try:
                        x, y = gaze_detector(frame)
                        if 0 <= x < self.width and 0 <= y < self.height:
                            self.gaze_points[point_idx].append((x, y))
                    except Exception as e:
                        logger.error(f"处理帧 {frame_idx} 时出错: {str(e)}")
                
                processed_frames += 1
                pbar.update(1)
        
        pbar.close()
        
        # 打印统计信息
        for i, points in enumerate(self.gaze_points):
            logger.info(f"标定点 {i+1}: 采集到 {len(points)} 个视线点")
        
        return self.gaze_points
    
    def compute_representative_points(self, method='kde'):
        """
        计算每个标定点的代表坐标
        
        参数：
            method: 计算方法 ('median', 'kde', 'weighted')
        """
        if not self.gaze_points:
            raise ValueError("请先运行 extract_gaze_points() 方法")
        
        logger.info("计算稳健代表点...")
        self.representative_points = []
        
        for point_idx, seg_points in enumerate(self.gaze_points):
            if not seg_points:
                logger.warning(f"标定点 {point_idx+1} 没有有效的视线点")
                self.representative_points.append((np.nan, np.nan))
                continue
            
            points = np.array(seg_points)
            x = points[:, 0]
            y = points[:, 1]
            
            if method == 'median':
                # 中位数方法
                rep_x = np.median(x)
                rep_y = np.median(y)
                
            elif method == 'kde':
                # 核密度估计峰值
                try:
                    kde = stats.gaussian_kde(points.T)
                    grid_x, grid_y = np.mgrid[0:self.width:100j, 0:self.height:100j]
                    positions = np.vstack([grid_x.ravel(), grid_y.ravel()])
                    density = kde(positions).reshape(grid_x.shape)
                    max_density_idx = np.argmax(density)
                    rep_x = grid_x.flatten()[max_density_idx]
                    rep_y = grid_y.flatten()[max_density_idx]
                except Exception as e:
                    logger.warning(f"核密度估计失败，使用中位数替代: {str(e)}")
                    rep_x = np.median(x)
                    rep_y = np.median(y)
                    
            else:  # 加权平均方法
                # 计算中位数作为初始参考点
                median_x = np.median(x)
                median_y = np.median(y)
                
                # 计算点到中位数的距离
                distances = np.sqrt((x - median_x)**2 + (y - median_y)**2)
                
                # 使用距离的倒数作为权重（距离越小权重越大）
                weights = 1.0 / (distances + 1e-5)  # 避免除零
                weights /= weights.sum()  # 归一化
                
                # 计算加权平均
                rep_x = np.sum(x * weights)
                rep_y = np.sum(y * weights)
            
            self.representative_points.append((rep_x, rep_y))
            theo_x, theo_y = self.actual_theoretical_points[point_idx]
            error = np.sqrt((rep_x - theo_x)**2 + (rep_y - theo_y)**2)
            logger.info(f"标定点 {point_idx+1}: 代表点({rep_x:.1f}, {rep_y:.1f}), "
                       f"理论点({theo_x}, {theo_y}), 误差={error:.1f}像素")
        
        return self.representative_points
    
    def train_mapping_model(self, model_type='tps'):
        """
        训练映射模型
        
        参数：
            model_type: 模型类型 ('tps' 或 'poly')
        """
        if not self.representative_points:
            raise ValueError("请先运行 compute_representative_points() 方法")
        
        # 获取有效点
        actual_points = []
        theoretical_points = []
        valid_indices = []
        
        for i, act in enumerate(self.representative_points):
            theo = self.theoretical_points[i]
            if not (np.isnan(act[0]) or np.isnan(act[1])):
                actual_points.append(act)
                theoretical_points.append(theo)
                valid_indices.append(i)
        
        actual_points = np.array(actual_points)
        theoretical_points = np.array(theoretical_points)
        
        if len(actual_points) < 4:
            raise ValueError(f"有效标定点不足 ({len(actual_points)})，至少需要4个点进行映射")
        
        logger.info(f"使用 {len(actual_points)} 个有效点训练 {model_type} 映射模型...")
        
        # 归一化坐标 (0-1范围)
        actual_points_norm = actual_points / np.array([self.width, self.height])
        theoretical_points_norm = theoretical_points  # 理论点已经是0-1范围
        
        # 训练模型
        self.mapping_type = model_type
        
        if model_type == 'tps':
            # 薄板样条插值
            self.mapping_model = RBFInterpolator(
                actual_points_norm, 
                theoretical_points_norm, 
                kernel='thin_plate_spline',
                smoothing=0.1
            )
        elif model_type == 'poly':
            # 多项式回归 (二阶)
            poly = PolynomialFeatures(degree=2)
            model = make_pipeline(poly, LinearRegression())
            model.fit(actual_points_norm, theoretical_points_norm)
            self.mapping_model = model
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 评估模型在训练集上的表现
        predicted = self.map_gaze_points(actual_points)
        mse = mean_squared_error(theoretical_points, predicted)
        logger.info(f"映射模型训练完成, 训练集 MSE: {mse:.6f}")
        
        return self.mapping_model
    
    def map_gaze_point(self, point):
        """
        映射单个视线点
        """
        if self.mapping_model is None:
            raise ValueError("请先训练映射模型")
        
        # 归一化输入点
        point_norm = np.array(point) / np.array([self.width, self.height])
        
        if self.mapping_type == 'tps':
            # TPS模型需要二维数组输入
            mapped = self.mapping_model(point_norm.reshape(1, -1))
            return mapped[0]
        else:
            # 多项式模型
            return self.mapping_model.predict(point_norm.reshape(1, -1))[0]
    
    def map_gaze_points(self, points):
        """
        映射多个视线点
        """
        if self.mapping_model is None:
            raise ValueError("请先训练映射模型")
        
        # 归一化输入点
        points_norm = np.array(points) / np.array([self.width, self.height])
        
        if self.mapping_type == 'tps':
            return self.mapping_model(points_norm)
        else:
            return self.mapping_model.predict(points_norm)
    
    def visualize_results(self, output_dir='results'):
        """
        可视化结果 (针对Linux环境优化中文显示)
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置Linux环境下中文显示
        try:
            # 尝试查找系统中可用的中文字体
            linux_chinese_fonts = [
                'WenQuanYi Micro Hei',  # 文泉驿微米黑
                'WenQuanYi Zen Hei',    # 文泉驿正黑
                'Noto Sans CJK SC',     # Google Noto Sans
                'Noto Sans CJK TC',
                'Droid Sans Fallback',  # Android字体
                'AR PL UMing CN',       # 文鼎明体
                'AR PL UKai CN',        # 文鼎楷体
                'DejaVu Sans',          # 备选字体
                'sans-serif'            # 最后回退
            ]
            
            # 检查系统可用字体
            available_fonts = set([f.name for f in mpl.font_manager.fontManager.ttflist])
            
            # 选择第一个可用的中文字体
            selected_font = None
            for font in linux_chinese_fonts:
                if font in available_fonts:
                    selected_font = font
                    break
            
            if selected_font:
                # 设置全局字体
                plt.rcParams['font.family'] = selected_font
                plt.rcParams['font.sans-serif'] = [selected_font]
                logger.info(f"使用中文字体: {selected_font}")
            else:
                # 如果没有找到中文字体，尝试添加字体路径
                logger.warning("未找到系统中文字体，尝试使用默认字体")
                plt.rcParams['font.family'] = 'sans-serif'
            
            # 确保能处理负号
            plt.rcParams['axes.unicode_minus'] = False
            
        except Exception as e:
            logger.warning(f"设置中文字体失败: {str(e)}")
        
        # 1. 绘制标定段位置
        plt.figure(figsize=(12, 6))
        if self.calibration_segments:
            for point_idx, start, end in self.calibration_segments:
                plt.plot(start, point_idx, 'go')  # 开始帧
                plt.plot(end, point_idx, 'ro')    # 结束帧
                plt.text(start, point_idx+0.1, f'P{point_idx}', fontsize=8)
        
        plt.title(f"基于时间的标定段划分 ({len(self.calibration_segments)}个标定段)")
        plt.xlabel("帧号")
        plt.ylabel("点索引")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_based_segments.png'))
        
        # 2. 绘制原始视线点
        plt.figure(figsize=(10, 8))
        for i, seg_points in enumerate(self.gaze_points):
            if seg_points:
                points = np.array(seg_points)
                plt.scatter(points[:, 0], points[:, 1], s=10, alpha=0.3, 
                        label=f'标定点 {i+1}' if i < 5 else None)
        
        # 绘制理论点位置
        for i, (x, y) in enumerate(self.actual_theoretical_points):
            plt.scatter(x, y, s=100, marker='x', color='red', label='理论点' if i == 0 else None)
        
        plt.title(f"原始视线落点")
        plt.xlabel("X坐标")
        plt.ylabel("Y坐标")
        plt.xlim(0, self.width)
        plt.ylim(self.height, 0)  # 反转Y轴以匹配图像坐标
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'raw_gaze_points.png'))
        
        # 3. 绘制代表点与理论点
        plt.figure(figsize=(10, 8))
        valid_actual = []
        valid_theoretical = []
        
        for i, act in enumerate(self.representative_points):
            theo = self.actual_theoretical_points[i]
            
            if not (np.isnan(act[0]) or np.isnan(act[1])):
                # 绘制理论点
                plt.scatter(theo[0], theo[1], s=150, 
                        marker='o', color='blue', edgecolors='black', label='理论点' if i == 0 else None)
                
                # 绘制代表点
                plt.scatter(act[0], act[1], s=100, marker='*', 
                        color='gold', edgecolors='black', label='代表点' if i == 0 else None)
                
                # 绘制连接线
                plt.plot([act[0], theo[0]], 
                        [act[1], theo[1]], 
                        'r--', linewidth=1, label='映射线' if i == 0 else None)
                
                valid_actual.append(act)
                valid_theoretical.append(theo)
                
                # 添加标签
                plt.text(act[0]+10, act[1]-10, f'{i+1}', fontsize=9, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        plt.title("代表点与理论点对比")
        plt.xlabel("X坐标")
        plt.ylabel("Y坐标")
        plt.xlim(0, self.width)
        plt.ylim(self.height, 0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'points_comparison.png'))
        
        # 4. 映射误差分析
        if valid_actual and self.mapping_model:
            valid_actual = np.array(valid_actual)
            valid_theoretical_norm = np.array(valid_theoretical) / np.array([self.width, self.height])
            
            # 计算映射误差
            mapped_points = self.map_gaze_points(valid_actual)
            errors = np.sqrt(np.sum((mapped_points - valid_theoretical_norm)**2, axis=1)) * 100  # 百分比误差
            
            plt.figure(figsize=(12, 5))
            plt.subplot(121)
            plt.bar(range(len(errors)), errors)
            plt.title("各标定点映射误差")
            plt.xlabel("标定点索引")
            plt.ylabel("误差(%)")
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.subplot(122)
            plt.hist(errors, bins=20, color='skyblue', edgecolor='black')
            plt.title("映射误差分布")
            plt.xlabel("误差(%)")
            plt.ylabel("频数")
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'mapping_error.png'))
            
            # 输出误差统计
            logger.info(f"映射误差统计: 平均={np.mean(errors):.2f}%, 中位数={np.median(errors):.2f}%, "
                        f"最大值={np.max(errors):.2f}%, 最小值={np.min(errors):.2f}%")
        
        plt.close('all')
    
    def process_full_calibration(self, gaze_detector, output_dir='results', 
                               sample_rate=1):
        """
        完整处理流程 (基于时间分割)
        """
        logger.info("开始完整标定流程 (基于时间分割)...")
        
        # 步骤1: 计算基于时间的标定段
        self.calculate_time_based_segments()
        
        if not self.calibration_segments:
            logger.error("无法计算标定段，无法继续处理")
            return False
        
        # 步骤2: 提取视线点
        self.extract_gaze_points(gaze_detector, sample_rate=sample_rate)
        
        # 步骤3: 计算代表点
        self.compute_representative_points()
        
        # 步骤4: 训练映射模型
        try:
            self.train_mapping_model('tps')
        except Exception as e:
            logger.error(f"训练映射模型失败: {str(e)}")
            return False
        
        # 步骤5: 可视化结果
        self.visualize_results(output_dir)
        
        # 保存结果
        results = {
            'calibration_segments': self.calibration_segments,
            'gaze_points': self.gaze_points,
            'representative_points': self.representative_points,
            'theoretical_points': self.theoretical_points,
            'mapping_type': self.mapping_type,
            'video_info': {
                'width': self.width,
                'height': self.height,
                'fps': self.fps,
                'total_frames': self.total_frames,
                'frame_duration_ms': self.frame_duration_ms
            }
        }
        
        np.save(os.path.join(output_dir, 'calibration_results.npy'), results)
        logger.info(f"结果已保存到 {output_dir} 目录")
        return True
    
    def close(self):
        """释放视频资源"""
        if self.cap.isOpened():
            self.cap.release()
        logger.info("视频资源已释放")


# 示例使用代码
if __name__ == "__main__":
    # 标定点的理论位置 (根据HTML中的布局生成)
    # 16点网格 + 中心点 (x, y 在0-1范围内)
    grid_size = 4
    margin = 0.01  # 留有一点点边界距离
    theoretical_points = []
    
    # 生成16个网格点
    for i in range(grid_size):
        for j in range(grid_size):
            x = margin + (1 - 2*margin) * (j / (grid_size - 1))
            y = margin + (1 - 2*margin) * (i / (grid_size - 1))
            theoretical_points.append((x, y))
    
    # 添加中心点
    theoretical_points.append((0.5, 0.5))
    
    # 模拟的视线检测函数 (实际项目中应替换为真实的视线检测模块)
    def mock_gaze_detector(frame):
        """模拟的视线落点检测函数"""
        # 在实际应用中，这里应调用您的视线检测模块
        # 返回随机点作为示例
        h, w = frame.shape[:2]
        return np.random.randint(0, w), np.random.randint(0, h)
    
    from gaze_point_for_cali import estimate_gaze
    
    # 创建分析器
    analyzer = VideoCalibrationAnalyzer('/home/public/RT_GENE/rt_gene/rt_gene_standalone/0810_calibration/test3.mp4', theoretical_points)
    
    try:
        # 运行完整处理流程
        success = analyzer.process_full_calibration(
            gaze_detector=estimate_gaze,
            output_dir='calibration_results',
            sample_rate=1        # 采样率
        )
        
        if success:
            # 测试映射函数
            test_point = (analyzer.width//2, analyzer.height//2)
            mapped_point = analyzer.map_gaze_point(test_point)
            logger.info(f"测试点映射: ({test_point[0]}, {test_point[1]}) -> ({mapped_point[0]:.4f}, {mapped_point[1]:.4f})")
        else:
            logger.error("标定流程失败，请检查日志了解详情")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
    finally:
        analyzer.close()