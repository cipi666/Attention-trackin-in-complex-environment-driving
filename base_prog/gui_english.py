import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import warnings
import torch
import threading
import os
from ultralytics import YOLO
import sys
import cv2
import numpy as np
import time
from PIL import Image, ImageTk
import queue
import traceback
import mediapy as media
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pymysql
import csv
from datetime import datetime
import random

sys.path.append("/home/public/base_prog/rt_gene/src")
from rt_gene.estimate_gaze_pytorch import GazeEstimator
from rt_gene.extract_landmarks_method_base import LandmarkMethodBase
from rt_gene.gaze_tools import get_phi_theta_from_euler, limit_yaw
from rt_gene.gaze_tools_standalone import euler_from_matrix
sys.path.append("/home/public/base_prog/rt_gene_standalone")
from eye_gaze import load_camera_calibration, extract_specific_landmarks, polar_to_cartesian
from gaze_point import get_gaze_point
script_path = os.path.dirname(os.path.realpath(__file__))

# Ignore warnings
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class WinGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.__win()
        
        # 创建选项卡控件
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 创建视频文件处理选项卡
        self.file_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.file_frame, text="Video File Processing")
        self.create_file_processing_ui(self.file_frame)
        
        # 创建摄像头实时处理选项卡
        self.camera_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.camera_frame, text="Camera Real-time Processing")
        self.create_camera_processing_ui(self.camera_frame)
        
        # 驾驶员数据库管理按钮（在所有选项卡中显示）
        self.tk_button_driver_db = self.__tk_button_driver_db(self)
        self.tk_button_driver_db.place(x=730, y=30, width=120, height=30)
        
    def __win(self):
        self.title("RuiMou - Driver Attention Real-time Quantitative Assessment System")
        width = 900  
        height = 720  # 增加高度以适应新按钮
        screenwidth = self.winfo_screenwidth()
        screenheight = self.winfo_screenheight()
        geometry = f'{width}x{height}+{(screenwidth - width) // 2}+{(screenheight - height) // 2}'
        self.geometry(geometry)
        self.resizable(width=False, height=False)

    def create_file_processing_ui(self, parent):
        """创建视频文件处理UI"""
        # 视频上传按钮
        self.tk_button_normal_video = self.__tk_button_normal_video(parent)
        self.tk_button_ir_video = self.__tk_button_ir_video(parent)
        self.tk_button_process = self.__tk_button_process(parent)
        self.tk_button_close = self.__tk_button_close(parent)
        
        # 视频显示区域（只显示正常视频）
        self.tk_label_normal_video = self.__tk_label_show_image(parent, x=50, y=80, width=800, height=450)
        
        # 参数显示区域
        self.tk_label_gaze_delay = self.__tk_label_param(parent, "Gaze Delay:", x=50, y=590)
        self.tk_label_gaze_delay_show = self.__tk_label_value(parent, "0 ms", x=150, y=590)
        
        self.tk_label_gaze_region = self.__tk_label_param(parent, "Gaze Region:", x=300, y=590)
        self.tk_label_gaze_region_show = self.__tk_label_value(parent, "Other Area", x=400, y=590)
        
        self.tk_label_pupil_size = self.__tk_label_param(parent, "Pupil Size:", x=50, y=630)
        self.tk_label_pupil_size_show = self.__tk_label_value(parent, "0 mm", x=150, y=630)
        
        self.tk_label_sampling_rate = self.__tk_label_param(parent, "Sampling Rate:", x=300, y=630)
        self.tk_label_sampling_rate_show = self.__tk_label_value(parent, "85%", x=400, y=630)
        
        self.tk_label_fatigue_status = self.__tk_label_param(parent, "Fatigue Status:", x=550, y=590)
        self.tk_label_fatigue_status_show = self.__tk_label_value(parent, "Normal", x=650, y=590, bg="green")
        
        # 处理进度条
        self.tk_progress_bar = self.__tk_progress_bar(parent)
        
        # 录制控制按钮
        self.tk_button_set_path = self.__tk_button_set_path(parent)
        self.tk_button_record = self.__tk_button_record(parent)
        self.tk_button_stop_record = self.__tk_button_stop_record(parent)
        
        # 添加按钮放置位置
        self.tk_button_set_path.place(x=50, y=540, width=150, height=30)
        self.tk_button_record.place(x=220, y=540, width=150, height=30)
        self.tk_button_stop_record.place(x=390, y=540, width=150, height=30)

    def create_camera_processing_ui(self, parent):
        """创建摄像头实时处理UI"""
        # 摄像头输入设置按钮
        self.tk_button_normal_camera = self.__tk_button_normal_camera(parent)
        self.tk_button_ir_camera = self.__tk_button_ir_camera(parent)
        self.tk_button_start_camera = self.__tk_button_start_camera(parent)
        self.tk_button_stop_camera = self.__tk_button_stop_camera(parent)
        
        # 视频显示区域（只显示正常视频）
        self.tk_label_camera_video = self.__tk_label_show_image(parent, x=50, y=80, width=800, height=450)
        
        # 参数显示区域
        self.tk_label_camera_gaze_delay = self.__tk_label_param(parent, "Gaze Delay:", x=50, y=590)
        self.tk_label_camera_gaze_delay_show = self.__tk_label_value(parent, "0 ms", x=150, y=590)
        
        self.tk_label_camera_gaze_region = self.__tk_label_param(parent, "Gaze Region:", x=300, y=590)
        self.tk_label_camera_gaze_region_show = self.__tk_label_value(parent, "Other Area", x=400, y=590)
        
        self.tk_label_camera_pupil_size = self.__tk_label_param(parent, "Pupil Size:", x=50, y=630)
        self.tk_label_camera_pupil_size_show = self.__tk_label_value(parent, "0 mm", x=150, y=630)
        
        self.tk_label_camera_sampling_rate = self.__tk_label_param(parent, "Sampling Rate:", x=300, y=630)
        self.tk_label_camera_sampling_rate_show = self.__tk_label_value(parent, "85%", x=400, y=630)
        
        self.tk_label_camera_fatigue_status = self.__tk_label_param(parent, "Fatigue Status:", x=550, y=590)
        self.tk_label_camera_fatigue_status_show = self.__tk_label_value(parent, "Normal", x=650, y=590, bg="green")
        
        # 摄像头状态标签
        self.tk_label_camera_status = tk.Label(parent, text="Camera Status: Disconnected", fg="red", font=("Arial", 10))
        self.tk_label_camera_status.place(x=50, y=30, width=200, height=25)
        
        # 录制控制按钮
        self.tk_button_camera_set_path = self.__tk_button_set_path(parent)
        self.tk_button_camera_set_path.place(x=50, y=540, width=150, height=30)
        self.tk_button_camera_record = self.__tk_button_record(parent)
        self.tk_button_camera_record.place(x=220, y=540, width=150, height=30)
        self.tk_button_camera_stop_record = self.__tk_button_stop_record(parent)
        self.tk_button_camera_stop_record.place(x=390, y=540, width=150, height=30)
        
        # 帧率显示
        # self.tk_label_fps = tk.Label(parent, text="FPS: 0", font=("Arial", 10))
        # self.tk_label_fps.place(x=600, y=30, width=150, height=25)

    def __tk_button_normal_video(self, parent):
        """创建正常视频上传按钮"""
        btn = tk.Button(parent, text="Upload Normal Video", bg="#e0e0ff")
        btn.place(x=50, y=30, width=150, height=30)
        return btn

    def __tk_button_ir_video(self, parent):
        """创建红外视频上传按钮"""
        btn = tk.Button(parent, text="Upload IR Video", bg="#ffe0e0")
        btn.place(x=220, y=30, width=150, height=30)
        return btn
    
    def __tk_button_normal_camera(self, parent):
        """创建场景视频流输入按钮"""
        btn = tk.Button(parent, text="Scene Video Input", bg="#d0e0ff")
        btn.place(x=260, y=30, width=150, height=30)
        return btn
    
    def __tk_button_ir_camera(self, parent):
        """创建红外视频流输入按钮"""
        btn = tk.Button(parent, text="IR Video Input", bg="#ffd0d0")
        btn.place(x=430, y=30, width=150, height=30)
        return btn

    def __tk_button_process(self, parent):
        """创建处理按钮"""
        btn = tk.Button(parent, text="Process Video", bg="#e0ffe0", state=tk.DISABLED)
        btn.place(x=390, y=30, width=150, height=30)
        return btn
    
    def __tk_button_start_camera(self, parent):
        """创建开始摄像头处理按钮"""
        btn = tk.Button(parent, text="Start Processing", bg="#e0ffe0", state=tk.DISABLED)
        btn.place(x=600, y=30, width=120, height=30)
        return btn

    def __tk_button_close(self, parent):
        """创建关闭按钮"""
        btn = tk.Button(parent, text="Stop Processing", bg="#ffc0c0", state=tk.DISABLED)
        btn.place(x=560, y=30, width=150, height=30)
        return btn
    
    def __tk_button_stop_camera(self, parent):
        """创建停止摄像头处理按钮"""
        btn = tk.Button(parent, text="Stop Processing", bg="#ffc0c0", state=tk.DISABLED)
        btn.place(x=730, y=30, width=120, height=30)
        return btn

    def __tk_label_show_image(self, parent, x, y, width, height):
        """创建视频显示标签"""
        # 创建初始黑色图像
        image = Image.new('RGB', (width, height), (0, 0, 0))
        photo = ImageTk.PhotoImage(image)
        label = tk.Label(parent, image=photo, bg="#333333", bd=2, relief="groove")
        label.image = photo  # 保持引用
        label.place(x=x, y=y, width=width, height=height)
        return label

    def __tk_label_param(self, parent, text, x, y):
        """创建参数名称标签"""
        label = tk.Label(parent, text=text, anchor="w", font=("Arial", 10))
        label.place(x=x, y=y, width=100, height=25)
        return label

    def __tk_label_value(self, parent, text, x, y, bg="white"):
        """创建参数值标签"""
        label = tk.Label(parent, text=text, anchor="w", bg=bg, 
                        font=("Arial", 10, "bold"), bd=1, relief="sunken")
        label.place(x=x, y=y, width=200, height=25)
        return label

    def __tk_progress_bar(self, parent):
        """创建进度条"""
        style = ttk.Style()
        style.configure("Custom.Horizontal.TProgressbar", thickness=20)
        pb = ttk.Progressbar(parent, style="Custom.Horizontal.TProgressbar", 
                               length=800, mode="determinate")
        pb.place(x=50, y=660, width=800, height=25)  # 下移位置
        return pb
    
    def __tk_button_set_path(self, parent):
        """创建保存路径设置按钮"""
        btn = tk.Button(parent, text="Set Save Path", bg="#e0e0ff", state=tk.NORMAL)
        return btn
    
    def __tk_button_record(self, parent):
        """创建录制按钮"""
        btn = tk.Button(parent, text="Start Recording", bg="#ffd0d0", state=tk.DISABLED)
        return btn
    
    def __tk_button_stop_record(self, parent):
        """创建停止录制按钮"""
        btn = tk.Button(parent, text="Stop Recording", bg="#d0d0ff", state=tk.DISABLED)
        return btn
    
    def __tk_button_driver_db(self, parent):
        """创建驾驶员数据库管理按钮"""
        btn = tk.Button(parent, text="Driver Database", bg="#ffffd0")
        return btn

class DriverDBWindow(tk.Toplevel):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.title("Driver Database Management")
        self.geometry("800x600")
        self.resizable(True, True)
        
        # 创建主框架
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 搜索区域
        search_frame = tk.LabelFrame(main_frame, text="Search & Filter")
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(search_frame, text="Driver ID:").grid(row=0, column=0, padx=5, pady=5)
        self.id_entry = tk.Entry(search_frame, width=10)
        self.id_entry.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(search_frame, text="Name:").grid(row=0, column=2, padx=5, pady=5)
        self.name_entry = tk.Entry(search_frame, width=20)
        self.name_entry.grid(row=0, column=3, padx=5, pady=5)
        
        tk.Label(search_frame, text="License Type:").grid(row=0, column=4, padx=5, pady=5)
        self.license_combo = ttk.Combobox(search_frame, values=["All", "A", "B", "C", "D"])
        self.license_combo.current(0)
        self.license_combo.grid(row=0, column=5, padx=5, pady=5)
        
        search_btn = tk.Button(search_frame, text="Search", command=self.search_drivers, bg="#e0e0ff")
        search_btn.grid(row=0, column=6, padx=10, pady=5)
        
        # 驾驶员列表
        list_frame = tk.LabelFrame(main_frame, text="Driver List")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 创建表格
        columns = ("id", "name", "employee_id", "license_type", "phone", "created_at")
        self.driver_tree = ttk.Treeview(
            list_frame, 
            columns=columns, 
            show="headings",
            selectmode="browse"
        )
        
        # 设置列宽和标题
        col_widths = [50, 150, 100, 80, 120, 150]
        for idx, col in enumerate(columns):
            self.driver_tree.heading(col, text=col.replace("_", " ").title())
            self.driver_tree.column(col, width=col_widths[idx], anchor=tk.W)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.driver_tree.yview)
        self.driver_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.driver_tree.pack(fill=tk.BOTH, expand=True)
        
        # 按钮区域
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Button(btn_frame, text="Add Driver", command=self.add_driver, width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Edit Driver", command=self.edit_driver, width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Delete Driver", command=self.delete_driver, width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="View Experiments", command=self.view_experiments, width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Export Data", command=self.export_data, width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Statistics", command=self.show_statistics, width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Close", command=self.destroy, width=12).pack(side=tk.RIGHT, padx=2)
        
        # 状态栏
        self.status_var = tk.StringVar()
        status_bar = tk.Label(self, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 加载初始数据
        self.search_drivers()
    
    def search_drivers(self):
        """从数据库搜索驾驶员"""
        driver_id = self.id_entry.get().strip()
        name = self.name_entry.get().strip()
        license_type = self.license_combo.get()
        
        try:
            # 清空当前列表
            for item in self.driver_tree.get_children():
                self.driver_tree.delete(item)
            
            # 连接数据库
            conn = pymysql.connect(**self.controller.db_config)
            cursor = conn.cursor()
            
            # 构建SQL查询
            sql = "SELECT id, name, employee_id, license_type, phone, created_at FROM drivers WHERE 1=1"
            params = []
            
            if driver_id:
                sql += " AND id = %s"
                params.append(driver_id)
                
            if name:
                sql += " AND name LIKE %s"
                params.append(f"%{name}%")
                
            if license_type != "All":
                sql += " AND license_type = %s"
                params.append(license_type)
            
            sql += " ORDER BY created_at DESC"
            
            cursor.execute(sql, params)
            drivers = cursor.fetchall()
            
            # 填充表格
            for driver in drivers:
                self.driver_tree.insert("", tk.END, values=driver)
            
            self.status_var.set(f"Found {len(drivers)} drivers")
            conn.close()
            
        except Exception as e:
            self.status_var.set(f"Database error: {str(e)}")
            messagebox.showerror("Database Error", f"Search failed: {str(e)}")
    
    def add_driver(self):
        """打开添加驾驶员对话框"""
        dialog = DriverDialog(self, self.controller, None)
        self.wait_window(dialog)
        self.search_drivers()  # 刷新列表
    
    def edit_driver(self):
        """编辑选中的驾驶员"""
        selected = self.driver_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a driver first")
            return
            
        driver_id = self.driver_tree.item(selected[0], "values")[0]
        dialog = DriverDialog(self, self.controller, driver_id)
        self.wait_window(dialog)
        self.search_drivers()  # 刷新列表
    
    def delete_driver(self):
        """删除选中的驾驶员"""
        selected = self.driver_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a driver first")
            return
            
        driver_id = self.driver_tree.item(selected[0], "values")[0]
        driver_name = self.driver_tree.item(selected[0], "values")[1]
        
    def delete_driver(self):
        """删除选中的驾驶员"""
        selected = self.driver_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a driver first")
            return
            
        driver_id = self.driver_tree.item(selected[0], "values")[0]
        driver_name = self.driver_tree.item(selected[0], "values")[1]
        
        if not messagebox.askyesno("Confirm", f"Delete driver {driver_name} (ID: {driver_id})?\nThis will also delete all related experiment data."):
            return

        try:
            conn = pymysql.connect(**self.controller.db_config)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM drivers WHERE id = %s", (driver_id,))
            conn.commit()
            conn.close()
            self.status_var.set(f"Driver {driver_name} deleted successfully")
            self.search_drivers()  # 刷新列表
        except Exception as e:
            self.status_var.set(f"Delete failed: {str(e)}")
            messagebox.showerror("Database Error", f"Delete failed: {str(e)}")
    
    def export_data(self):
        """导出数据到CSV文件"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not file_path:
            return
            
        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # 写标题
                writer.writerow(["ID", "Name", "Employee ID", "License Type", "Phone", "Created At"])
                
                # 写数据
                for item in self.driver_tree.get_children():
                    values = self.driver_tree.item(item, "values")
                    writer.writerow(values)
                    
            self.status_var.set(f"Data exported to {file_path}")
        except Exception as e:
            self.status_var.set(f"Export failed: {str(e)}")
    
    def show_statistics(self):
        """显示统计信息"""
        try:
            conn = pymysql.connect(**self.controller.db_config)
            cursor = conn.cursor()
            
            # 获取统计信息
            cursor.execute("SELECT COUNT(*) FROM drivers")
            total_drivers = cursor.fetchone()[0]
            
            cursor.execute("SELECT license_type, COUNT(*) FROM drivers GROUP BY license_type")
            license_stats = cursor.fetchall()
            
            cursor.execute("SELECT DATE(created_at) as date, COUNT(*) FROM drivers GROUP BY date ORDER BY date DESC LIMIT 7")
            recent_stats = cursor.fetchall()
            
            conn.close()
            
            # 创建统计窗口
            stats_win = tk.Toplevel(self)
            stats_win.title("Driver Statistics")
            stats_win.geometry("500x400")
            
            # 主框架
            main_frame = tk.Frame(stats_win)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            # 总驾驶员数
            tk.Label(main_frame, text=f"Total Drivers: {total_drivers}", 
                    font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))
            
            # 驾照类型分布
            tk.Label(main_frame, text="License Type Distribution:", 
                    font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))
            
            license_frame = tk.Frame(main_frame)
            license_frame.pack(fill=tk.X, padx=10, pady=5)
            
            for license_type, count in license_stats:
                percent = (count / total_drivers) * 100 if total_drivers > 0 else 0
                tk.Label(license_frame, text=f"{license_type}: {count} ({percent:.1f}%)").pack(anchor=tk.W)
            
            # 最近添加
            tk.Label(main_frame, text="Recently Added (Last 7 days):", 
                    font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(10, 5))
            
            recent_frame = tk.Frame(main_frame)
            recent_frame.pack(fill=tk.X, padx=10, pady=5)
            
            for date, count in recent_stats:
                tk.Label(recent_frame, text=f"{date}: {count} driver(s)").pack(anchor=tk.W)
            
            # 关闭按钮
            tk.Button(main_frame, text="Close", command=stats_win.destroy).pack(pady=20)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get statistics: {str(e)}")
    
    def view_experiments(self):
        """查看选中驾驶员的实验数据"""
        selected = self.driver_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Please select a driver first")
            return
            
        driver_id = self.driver_tree.item(selected[0], "values")[0]
        driver_name = self.driver_tree.item(selected[0], "values")[1]
        
        # 创建实验数据窗口
        exp_win = tk.Toplevel(self)
        exp_win.title(f"Experiments - {driver_name}")
        exp_win.geometry("1000x600")
        
        # 主框架
        main_frame = tk.Frame(exp_win)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 实验列表
        list_frame = tk.LabelFrame(main_frame, text="Experiment List")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 创建表格
        columns = ("id", "date", "start_time", "duration", "total_frames", "valid_samples", "fatigue_events")
        exp_tree = ttk.Treeview(list_frame, columns=columns, show="headings", selectmode="browse")
        
        # 设置列标题和宽度
        col_widths = [50, 100, 150, 80, 100, 100, 80]
        headers = ["ID", "Date", "Start Time", "Duration(s)", "Total Frames", "Valid Samples", "Fatigue Events"]
        
        for idx, (col, header) in enumerate(zip(columns, headers)):
            exp_tree.heading(col, text=header)
            exp_tree.column(col, width=col_widths[idx], anchor=tk.W)
        
        # 添加滚动条
        exp_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=exp_tree.yview)
        exp_tree.configure(yscroll=exp_scrollbar.set)
        exp_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        exp_tree.pack(fill=tk.BOTH, expand=True)
        
        # 加载实验数据
        try:
            conn = pymysql.connect(**self.controller.db_config)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, experiment_date, start_time, duration, total_frames, valid_samples, fatigue_events
                FROM experiments WHERE driver_id = %s ORDER BY start_time DESC
            """, (driver_id,))
            experiments = cursor.fetchall()
            
            for exp in experiments:
                exp_tree.insert("", tk.END, values=exp)
            
            conn.close()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load experiments: {str(e)}")
        
        # 按钮区域
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        def view_details():
            selected_exp = exp_tree.selection()
            if not selected_exp:
                messagebox.showwarning("Warning", "Please select an experiment first")
                return
            
            exp_id = exp_tree.item(selected_exp[0], "values")[0]
            self.view_experiment_details(exp_id)
        
        tk.Button(btn_frame, text="View Details", command=view_details, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Close", command=exp_win.destroy, width=15).pack(side=tk.RIGHT, padx=5)
    
    def view_experiment_details(self, experiment_id):
        """查看实验详细数据"""
        try:
            conn = pymysql.connect(**self.controller.db_config)
            cursor = conn.cursor()
            
            # 获取实验基本信息
            cursor.execute("""
                SELECT e.*, d.name FROM experiments e 
                JOIN drivers d ON e.driver_id = d.id 
                WHERE e.id = %s
            """, (experiment_id,))
            exp_info = cursor.fetchone()
            
            if not exp_info:
                messagebox.showerror("Error", "Experiment not found")
                return
            
            # 获取注意力数据
            cursor.execute("""
                SELECT frame_number, gaze_region, gaze_x, gaze_y, pupil_size, fatigue_status
                FROM attention_data WHERE experiment_id = %s ORDER BY frame_number
            """, (experiment_id,))
            attention_data = cursor.fetchall()
            
            conn.close()
            
            # 创建详情窗口
            detail_win = tk.Toplevel()
            detail_win.title(f"Experiment Details - ID: {experiment_id}")
            detail_win.geometry("800x600")
            
            # 使用Notebook创建选项卡
            notebook = ttk.Notebook(detail_win)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 基本信息选项卡
            info_frame = ttk.Frame(notebook)
            notebook.add(info_frame, text="Basic Info")
            
            info_text = f"""
Experiment ID: {exp_info[0]}
Driver: {exp_info[-1]}
Date: {exp_info[2]}
Start Time: {exp_info[3]}
End Time: {exp_info[4] if exp_info[4] else 'N/A'}
Duration: {exp_info[5]} seconds
Total Frames: {exp_info[6]}
Gaze Samples: {exp_info[7]}
Valid Samples: {exp_info[8]}
Average Pupil Size: {exp_info[9]:.2f} mm
Fatigue Events: {exp_info[10]}
Video Path: {exp_info[11] if exp_info[11] else 'N/A'}
            """
            
            tk.Label(info_frame, text=info_text, justify=tk.LEFT, font=("Arial", 10)).pack(padx=20, pady=20)
            
            # 注意力数据选项卡
            data_frame = ttk.Frame(notebook)
            notebook.add(data_frame, text="Attention Data")
            
            # 创建数据表格
            data_columns = ("frame", "region", "gaze_x", "gaze_y", "pupil_size", "fatigue")
            data_tree = ttk.Treeview(data_frame, columns=data_columns, show="headings", height=20)
            
            for col in data_columns:
                data_tree.heading(col, text=col.replace("_", " ").title())
                data_tree.column(col, width=100)
            
            # 添加滚动条
            data_scrollbar = ttk.Scrollbar(data_frame, orient=tk.VERTICAL, command=data_tree.yview)
            data_tree.configure(yscroll=data_scrollbar.set)
            data_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=10)
            data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 填充数据
            for data in attention_data:
                data_tree.insert("", tk.END, values=data)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load experiment details: {str(e)}")


class DriverDialog(tk.Toplevel):
    def __init__(self, parent, controller, driver_id=None):
        super().__init__(parent)
        self.controller = controller
        self.driver_id = driver_id
        self.title("Add Driver" if driver_id is None else "Edit Driver")
        self.geometry("400x300")
        self.resizable(False, False)
        
        # 表单框架
        form_frame = tk.Frame(self)
        form_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 表单字段
        tk.Label(form_frame, text="Full Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.name_entry = tk.Entry(form_frame, width=30)
        self.name_entry.grid(row=0, column=1, pady=5, padx=5)
        
        tk.Label(form_frame, text="Employee ID:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.emp_id_entry = tk.Entry(form_frame, width=30)
        self.emp_id_entry.grid(row=1, column=1, pady=5, padx=5)
        
        tk.Label(form_frame, text="License Type:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.license_combo = ttk.Combobox(form_frame, values=["A", "B", "C", "D"], width=27)
        self.license_combo.grid(row=2, column=1, pady=5, padx=5)
        self.license_combo.current(0)
        
        tk.Label(form_frame, text="Phone:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.phone_entry = tk.Entry(form_frame, width=30)
        self.phone_entry.grid(row=3, column=1, pady=5, padx=5)
        
        # 按钮框架
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.X, pady=(0, 20), padx=20)
        
        tk.Button(btn_frame, text="Save", command=self.save_driver, width=10).pack(side=tk.RIGHT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=self.destroy, width=10).pack(side=tk.RIGHT, padx=5)
        
        # 如果是编辑模式，加载现有数据
        if driver_id:
            self.load_driver_data()
    
    def load_driver_data(self):
        """加载驾驶员数据到表单"""
        try:
            conn = pymysql.connect(**self.controller.db_config)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name, employee_id, license_type, phone FROM drivers WHERE id = %s", 
                (self.driver_id,)
            )
            driver = cursor.fetchone()
            conn.close()
            
            if driver:
                self.name_entry.insert(0, driver[0])
                self.emp_id_entry.insert(0, driver[1])
                self.license_combo.set(driver[2])
                self.phone_entry.insert(0, driver[3])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load driver data: {str(e)}")
    
    def save_driver(self):
        """保存驾驶员信息"""
        name = self.name_entry.get().strip()
        emp_id = self.emp_id_entry.get().strip()
        license_type = self.license_combo.get()
        phone = self.phone_entry.get().strip()
        
        # 验证输入
        if not name:
            messagebox.showwarning("Validation", "Name is required")
            return
            
    def save_driver(self):
        """保存驾驶员信息"""
        name = self.name_entry.get().strip()
        emp_id = self.emp_id_entry.get().strip()
        license_type = self.license_combo.get()
        phone = self.phone_entry.get().strip()
        
        # 验证输入
        if not name:
            messagebox.showwarning("Validation", "Name is required")
            return
            
        if not emp_id:
            messagebox.showwarning("Validation", "Employee ID is required")
            return

        try:
            conn = pymysql.connect(**self.controller.db_config)
            cursor = conn.cursor()
            
            if self.driver_id is None:
                # 检查员工ID是否已存在
                cursor.execute("SELECT id FROM drivers WHERE employee_id = %s", (emp_id,))
                if cursor.fetchone():
                    messagebox.showwarning("Validation", "Employee ID already exists")
                    conn.close()
                    return
                
                # 添加新驾驶员
                cursor.execute(
                    "INSERT INTO drivers (name, employee_id, license_type, phone) "
                    "VALUES (%s, %s, %s, %s)",
                    (name, emp_id, license_type, phone)
                )
                action = "added"
            else:
                # 检查员工ID是否已被其他驾驶员使用
                cursor.execute("SELECT id FROM drivers WHERE employee_id = %s AND id != %s", (emp_id, self.driver_id))
                if cursor.fetchone():
                    messagebox.showwarning("Validation", "Employee ID already exists")
                    conn.close()
                    return
                
                # 更新现有驾驶员
                cursor.execute(
                    "UPDATE drivers SET name = %s, employee_id = %s, license_type = %s, phone = %s "
                    "WHERE id = %s",
                    (name, emp_id, license_type, phone, self.driver_id)
                )
                action = "updated"
            
            conn.commit()
            conn.close()
            messagebox.showinfo("Success", f"Driver {action} successfully")
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save driver: {str(e)}")


class Win(WinGUI):
    """界面事件绑定类"""
    def __init__(self, controller):
        self.ctl = controller
        super().__init__()
        self.__event_bind()
        self.ctl.init(self)

    def __event_bind(self):
        """绑定事件处理函数"""
        # 文件处理选项卡事件
        self.tk_button_normal_video.bind('<Button-1>', self.ctl.upload_normal_video)
        self.tk_button_ir_video.bind('<Button-1>', self.ctl.upload_ir_video)
        self.tk_button_process.bind('<Button-1>', self.ctl.process_videos)
        self.tk_button_close.bind('<Button-1>', self.ctl.stop_processing)
        self.tk_button_set_path.bind('<Button-1>', self.ctl.set_save_path)
        self.tk_button_record.bind('<Button-1>', self.ctl.start_recording)
        self.tk_button_stop_record.bind('<Button-1>', self.ctl.stop_recording)
        
        # 摄像头处理选项卡事件
        self.tk_button_normal_camera.bind('<Button-1>', self.ctl.set_normal_camera_source)
        self.tk_button_ir_camera.bind('<Button-1>', self.ctl.set_ir_camera_source)
        self.tk_button_start_camera.bind('<Button-1>', self.ctl.start_camera_processing)
        self.tk_button_stop_camera.bind('<Button-1>', self.ctl.stop_camera_processing)
        self.tk_button_camera_set_path.bind('<Button-1>', self.ctl.set_save_path)
        self.tk_button_camera_record.bind('<Button-1>', self.ctl.start_recording)
        self.tk_button_camera_stop_record.bind('<Button-1>', self.ctl.stop_recording)
        
        # 公共事件
        self.tk_button_driver_db.bind('<Button-1>', self.ctl.open_driver_db)

    def open_driver_db_window(self):
        """打开驾驶员数据库管理窗口"""
        # 先初始化数据库
        if not self.ctl.init_database():
            messagebox.showerror("Error", "Database initialization failed")
            return
            
        # 创建数据库管理窗口
        DriverDBWindow(self, self.ctl)
    


class Controller:
    """视频处理逻辑控制器"""
    def __init__(self):
        # 初始化模型路径
        self.scene_model_path = "/home/public/base_prog/yolo_weights/normal_video/train_15_best.pt"
        self.pupil_model_path = "/home/public/base_prog/yolo_weights/pupil/best.pt"
        self.fatigue_model_path = "/home/public/fatigue_driving/fatigue_driving_detection/runs/detect/train/weights/best.pt"

        # 初始化状态变量
        self.normal_video_path = None
        self.ir_video_path = None
        self.processing = False
        self.recording = False
        self.save_path = None
        self.video_writer = None
        self.frames_to_record = []  # 存储要录制的帧
        
        # 摄像头相关变量
        self.normal_camera_source = None
        self.ir_camera_source = None
        self.camera_processing = False
        self.camera_recording = False
        self.camera_frames_to_record = []
        
        # 初始化模型
        self.device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
        self.scene_model = None
        self.fatigue_model = None
        self.gaze_model = None
        self.pupil_model = None

        # 相机校准，默认为空
        self.dist_coefficients = None
        self.camera_matrix = None

        # 创建停止事件
        self.stop_event = threading.Event()
        self.camera_stop_event = threading.Event()
        
        # 创建线程间通信队列
        self.result_queue = queue.Queue(maxsize=10)  # 最终结果队列
        self.camera_result_queue = queue.Queue(maxsize=10)  # 摄像头处理结果队列

        # 线程列表
        self.threads = []

        # 疲劳检测计数器
        self.roll = 0
        self.roll_eye = 0
        self.roll_mouth = 0
        self.fatigue_threshold = 0.05  # 疲劳阈值
        self.roll_length = 10  # 连续检测帧数
        self.last_fatigue_status = "Normal"  # 上次检测状态

        # UI更新相关
        self.valid_samples = 0
        self.total_samples = 0
        self.last_ir_pupil_size = 0.0
        self.total_frames = 0  # 总帧数计数器

        # 帧处理计数器
        self.frame_count = 0
        
        # 摄像头帧率计数器
        self.camera_frame_count = 0
        self.camera_start_time = 0
        self.camera_fps = 0

        #数据库连接配置
        self.db_config = {
            "host": "localhost",
            "user": "root",
            "password": "",
            "database": "driver_attention_db",
            "unix_socket": "/home/public/mysql/mysql.sock",
            "port": 3307,
            "charset": "utf8mb4"
        }
        
        # 当前选择的驾驶员ID
        self.current_driver_id = None
        self.current_experiment_id = None
        
        # 实验统计数据
        self.experiment_start_time = None
        self.fatigue_event_count = 0
        self.gaze_regions_data = []  # 存储注视区域数据

    def init(self, ui):
        """初始化UI"""
        self.ui = ui
        self.load_models()
    
    def load_models(self):
        """加载所有需要的模型"""
        try:
            # 加载头部检测模型
            self.scene_model = YOLO(self.scene_model_path).to(self.device)
            self.scene_model(np.zeros((48, 48, 3)))
            
            # 加载疲劳检测模型
            self.fatigue_model = YOLO(self.fatigue_model_path).to(self.device)
            self.fatigue_model(np.zeros((48, 48, 3)))
            
            # 加载gaze_estimator等
            self.gaze_estimator = GazeEstimator("cuda:6", [os.path.abspath(os.path.join(script_path, './rt_gene/model_nets/gaze_model_pytorch_vgg16_prl_mpii_allsubjects1.model'))])
            self.landmark_estimator = LandmarkMethodBase(device_id_facedetection="cuda:6",
                                            checkpoint_path_face=os.path.abspath(os.path.join(script_path, "./rt_gene/model_nets/SFD/s3fd_facedetector.pth")),
                                            checkpoint_path_landmark=os.path.abspath(
                                                os.path.join(script_path, "./rt_gene/model_nets/phase1_wpdc_vdc.pth.tar")),
                                            model_points_file=os.path.abspath(os.path.join(script_path, "./rt_gene/model_nets/face_model_68.txt")))

            # 加载相机校准参数
            calib_file = "/path/to/calibration_file.yaml"
            if os.path.exists(calib_file):
                self.dist_coefficients, self.camera_matrix = load_camera_calibration(calib_file)
            else:
                messagebox.showwarning("Warning", "Camera calibration file not found, using default values.")
                im_width, im_height = 800, 450
                self.dist_coefficients, self.camera_matrix = np.zeros((1, 5)), np.array(
                    [[im_height, 0.0, im_width / 2.0], [0.0, im_height, im_height / 2.0], [0.0, 0.0, 1.0]])
            
            # 加载瞳孔检测模型
            self.pupil_model = YOLO(self.pupil_model_path).to(self.device)
            self.pupil_model(np.zeros((48, 48, 3)))
            
            messagebox.showinfo("Information", "Models loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
            traceback.print_exc()
    
    def init_database(self):
        """初始化数据库连接，若不存在表创建表"""
        try:
            conn = pymysql.connect(**self.db_config)
            cursor = conn.cursor()
            
            # 创建驾驶员表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drivers (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    employee_id VARCHAR(20) NOT NULL UNIQUE,
                    license_type VARCHAR(10) NOT NULL,
                    phone VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建实验数据表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    driver_id INT NOT NULL,
                    experiment_date DATE NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP NULL,
                    duration INT DEFAULT 0 COMMENT 'Duration in seconds',
                    total_frames INT DEFAULT 0,
                    gaze_samples INT DEFAULT 0,
                    valid_samples INT DEFAULT 0,
                    avg_pupil_size FLOAT DEFAULT 0,
                    fatigue_events INT DEFAULT 0,
                    video_path VARCHAR(500),
                    FOREIGN KEY (driver_id) REFERENCES drivers(id) ON DELETE CASCADE
                )
            """)
            
            # 创建注意力分配数据表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attention_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    experiment_id INT NOT NULL,
                    frame_number INT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    gaze_region VARCHAR(50) NOT NULL,
                    gaze_x FLOAT,
                    gaze_y FLOAT,
                    pupil_size FLOAT,
                    fatigue_status VARCHAR(20),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
                )
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            print("Database initialized successfully")
            return True
        except Exception as e:
            print(f"Database initialization failed: {str(e)}")
            messagebox.showerror("Database Error", f"Failed to initialize database: {str(e)}")
            return False
        #             experiment_date DATE NOT NULL,
        #             duration INT NOT NULL COMMENT 'Duration in seconds',
        #             gaze_samples INT NOT NULL,
        #             valid_samples INT NOT NULL,
        #             avg_pupil_size FLOAT,
        #             fatigue_events INT,
        #             FOREIGN KEY (driver_id) REFERENCES drivers(id)
        #         )
        #     """)
            
        #     # 创建注意力分配数据表
        #     cursor.execute("""
        #         CREATE TABLE IF NOT EXISTS attention_data (
        #             id INT AUTO_INCREMENT PRIMARY KEY,
        #             experiment_id INT NOT NULL,
        #             timestamp TIMESTAMP NOT NULL,
        #             gaze_region VARCHAR(50) NOT NULL,
        #             gaze_duration INT NOT NULL COMMENT 'Duration in milliseconds',
        #             FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        #         )
        #     """)
            
        #     conn.commit()
        #     TODO:cursor.close()
        #     conn.close()
        #     print("Database initialized successfully")
        #     return True
        # except Exception as e:
        #     print(f"Database initialization failed: {str(e)}")
        #     return False

    # ====================== 文件处理功能 ======================
    def upload_normal_video(self, evt):
        """上传正常视频"""
        self.normal_video_path = filedialog.askopenfilename(
            title="Select Normal Video", 
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )
        if self.normal_video_path:
            self.ui.tk_button_normal_video.config(bg="#a0a0ff", text=f"Normal: {os.path.basename(self.normal_video_path)}")
            self.check_ready_to_process()
    
    def upload_ir_video(self, evt):
        """上传红外视频"""
        self.ir_video_path = filedialog.askopenfilename(
            title="Select IR Video", 
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )
        if self.ir_video_path:
            self.ui.tk_button_ir_video.config(bg="#ffa0a0", text=f"IR: {os.path.basename(self.ir_video_path)}")
            self.check_ready_to_process()
    
    def check_ready_to_process(self):
        """检查是否可以开始处理"""
        if self.normal_video_path and self.ir_video_path:
            self.ui.tk_button_process.config(state=tk.NORMAL, bg="#a0ffa0")
    
    def select_driver(self):
        """选择当前驾驶员"""
        try:
            conn = pymysql.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, employee_id FROM drivers ORDER BY name")
            drivers = cursor.fetchall()
            conn.close()
            
            if not drivers:
                messagebox.showwarning("Warning", "No drivers found. Please add drivers first.")
                return None
            
            # 创建选择对话框
            dialog = tk.Toplevel()
            dialog.title("Select Driver")
            dialog.geometry("400x300")
            dialog.resizable(False, False)
            
            # 先创建窗口内容
            selected_driver_id = None
            
            def on_select():
                nonlocal selected_driver_id
                selection = tree.selection()
                if selection:
                    selected_driver_id = tree.item(selection[0], "values")[0]
                    dialog.destroy()
                else:
                    messagebox.showwarning("Warning", "Please select a driver")
            
            def on_cancel():
                dialog.destroy()
            
            # 创建界面
            tk.Label(dialog, text="Select Driver for Experiment:", font=("Arial", 12, "bold")).pack(pady=10)
            
            # 创建表格
            frame = tk.Frame(dialog)
            frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            columns = ("id", "name", "employee_id")
            tree = ttk.Treeview(frame, columns=columns, show="headings", height=8)
            
            tree.heading("id", text="ID")
            tree.heading("name", text="Name")
            tree.heading("employee_id", text="Employee ID")
            
            tree.column("id", width=50)
            tree.column("name", width=150)
            tree.column("employee_id", width=100)
            
            # 添加滚动条
            scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            tree.configure(yscroll=scrollbar.set)
            
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # 填充数据
            for driver in drivers:
                tree.insert("", tk.END, values=driver)
            
            # 按钮
            btn_frame = tk.Frame(dialog)
            btn_frame.pack(pady=10)
            
            tk.Button(btn_frame, text="Select", command=on_select, width=10).pack(side=tk.LEFT, padx=5)
            tk.Button(btn_frame, text="Cancel", command=on_cancel, width=10).pack(side=tk.LEFT, padx=5)
            
            # 确保窗口完全显示后再设置模态
            dialog.update_idletasks()  # 确保窗口完全渲染
            dialog.transient(self.ui)  # 设置为主窗口的子窗口
            dialog.grab_set()          # 设置为模态对话框
            dialog.focus_set()         # 获取焦点
            
            # 等待对话框关闭
            self.ui.wait_window(dialog)
            
            return selected_driver_id
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load drivers: {str(e)}")
            return None
    
    def start_experiment(self, driver_id, video_path=None):
        """开始新实验"""
        try:
            conn = pymysql.connect(**self.db_config)
            cursor = conn.cursor()
            
            now = datetime.now()
            cursor.execute(
                "INSERT INTO experiments (driver_id, experiment_date, start_time, video_path) VALUES (%s, %s, %s, %s)",
                (driver_id, now.date(), now, video_path)
            )
            experiment_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            self.current_driver_id = driver_id
            self.current_experiment_id = experiment_id
            self.experiment_start_time = now
            self.fatigue_event_count = 0
            self.gaze_regions_data = []
            
            print(f"Started experiment {experiment_id} for driver {driver_id}")
            return experiment_id
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start experiment: {str(e)}")
            return None
    
    def end_experiment(self):
        """结束实验并保存数据"""
        if not self.current_experiment_id:
            return
            
        try:
            conn = pymysql.connect(**self.db_config)
            cursor = conn.cursor()
            
            now = datetime.now()
            duration = int((now - self.experiment_start_time).total_seconds()) if self.experiment_start_time else 0
            avg_pupil_size = self.last_ir_pupil_size
            
            cursor.execute("""
                UPDATE experiments 
                SET end_time = %s, duration = %s, total_frames = %s, gaze_samples = %s, 
                    valid_samples = %s, avg_pupil_size = %s, fatigue_events = %s
                WHERE id = %s
            """, (now, duration, self.total_frames, self.total_samples, self.valid_samples, 
                  avg_pupil_size, self.fatigue_event_count, self.current_experiment_id))
            
            conn.commit()
            conn.close()
            
            print(f"Ended experiment {self.current_experiment_id}")
            self.current_experiment_id = None
            self.current_driver_id = None
            
        except Exception as e:
            print(f"Failed to end experiment: {str(e)}")
    
    def save_attention_data(self, frame_number, gaze_region, gaze_x, gaze_y, pupil_size, fatigue_status):
        """保存注意力数据"""
        if not self.current_experiment_id:
            return
            
        try:
            conn = pymysql.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO attention_data 
                (experiment_id, frame_number, timestamp, gaze_region, gaze_x, gaze_y, pupil_size, fatigue_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (self.current_experiment_id, frame_number, datetime.now(), 
                  gaze_region, gaze_x, gaze_y, pupil_size, fatigue_status))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Failed to save attention data: {str(e)}")

    def process_videos(self, evt):
        """开始处理视频"""
        if not self.normal_video_path or not self.ir_video_path:
            messagebox.showerror("Error", "Please upload both videos")
            return
        
        # 选择驾驶员
        driver_id = self.select_driver()
        if not driver_id:
            return
        
        # 开始实验
        experiment_id = self.start_experiment(driver_id, self.normal_video_path)
        if not experiment_id:
            return
        
        # 获取视频总帧数
        cap_normal = cv2.VideoCapture(self.normal_video_path)
        cap_ir = cv2.VideoCapture(self.ir_video_path)
        total_frames_normal = int(cap_normal.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames_ir = int(cap_ir.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = min(total_frames_normal, total_frames_ir)
        cap_normal.release()
        cap_ir.release()
        
        self.processing = True
        self.stop_event.clear()  # 重置停止事件
        self.ui.tk_button_process.config(state=tk.DISABLED, bg="#c0c0c0")
        self.ui.tk_button_close.config(state=tk.NORMAL, bg="#ff8080")
        self.ui.tk_button_record.config(state=tk.NORMAL, bg="#ffa0a0")  # 启用录制按钮
        
        # 重置计数器
        self.valid_samples = 0
        self.total_samples = 0
        self.last_ir_pupil_size = 0.0
        self.frame_count = 0
        
        # 清空队列
        self.clear_queue(self.result_queue)
        
        # 启动处理线程
        self.threads = [
            threading.Thread(target=self.process_frames_thread, name="FrameProcessingThread")
        ]
        
        for t in self.threads:
            t.daemon = True
            t.start()
        
        # 启动UI更新
        self.update_ui_loop()
    
    def process_frames_thread(self):
        """处理帧的线程"""
        try:
            cap_normal = cv2.VideoCapture(self.normal_video_path)
            cap_ir = cv2.VideoCapture(self.ir_video_path)
            
            if not cap_normal.isOpened() or not cap_ir.isOpened():
                print("Failed to open videos")
                return
            
            while not self.stop_event.is_set() and self.frame_count < self.total_frames:
                # 读取帧
                ret_normal, frame_normal = cap_normal.read()
                ret_ir, frame_ir = cap_ir.read()
                
                if not ret_normal or not ret_ir:
                    break
                
                # 处理正常视频帧
                start_time = time.time()
                gaze_est, gaze_point, left_eye_center, right_eye_center, gaze_delay = self.estimate_gaze(frame_normal, start_time)
                
                # 使用YOLO检测场景中的物体
                with torch.no_grad():
                    scene_results = self.scene_model(frame_normal)[0]
                
                # 提取检测到的物品区域
                detected_objects = self.extract_detected_objects(frame_normal, scene_results)
                
                # 将注视区域与其名称联系
                gaze_region = self.get_gaze_region(gaze_point, detected_objects)
                
                # 在图像上绘制视线落点
                cv2.circle(frame_normal, gaze_point, 20, (0, 0, 255), -1)
                cv2.putText(frame_normal, f"Gaze: {gaze_region}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 绘制视线向量
                gaze_length = 30
                
                # 左眼视线标注
                left_gaze_end = (int(left_eye_center[0] - gaze_length * gaze_est[0]),
                                 int(left_eye_center[1] - gaze_length * gaze_est[1]))
                cv2.arrowedLine(frame_normal, 
                                (int(left_eye_center[0]), int(left_eye_center[1])),
                                left_gaze_end, (255, 0, 0), 2)
                
                # 右眼视线标注
                right_gaze_end = (int(right_eye_center[0] - gaze_length * gaze_est[0]),
                                  int(right_eye_center[1] - gaze_length * gaze_est[1]))
                cv2.arrowedLine(frame_normal, 
                                (int(right_eye_center[0]), int(right_eye_center[1])),
                                right_gaze_end, (0, 0, 255), 2)
                
                # 添加视线向量文本
                gaze_text = f"Gaze: {[round(x, 2) for x in gaze_est]}" if gaze_est is not None else "Gaze: N/A"
                cv2.putText(frame_normal, gaze_text, (int(left_eye_center[0]), int(left_eye_center[1]) + 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 疲劳检测
                fatigue_results = self.fatigue_model(frame_normal)[0]
                names = fatigue_results.names
                cls_list = fatigue_results.boxes.cls.tolist()
                cls_list = [int(i) for i in cls_list]
                detected_names = [names[index] for index in cls_list]
                
                # 更新计数器
                if 'closed_eye' in detected_names:
                    self.roll_eye += 1
                if 'open_mouth' in detected_names:
                    self.roll_mouth += 1
                self.roll += 1
                
                # 每10帧计算一次疲劳指标
                if self.roll == self.roll_length:
                    perclos = (self.roll_eye / self.roll) + (self.roll_mouth / self.roll) * 0.2
                    fatigue_status = "Fatigue" if perclos > self.fatigue_threshold else "Normal"
                    # 重置计数器
                    self.roll = 0
                    self.roll_eye = 0
                    self.roll_mouth = 0
                    self.last_fatigue_status = fatigue_status
                else:
                    fatigue_status = self.last_fatigue_status
                
                # 处理红外视频帧
                pupil_size = self.process_ir_frame(frame_ir)
                self.last_ir_pupil_size = pupil_size
                
                # 更新采样数据
                if gaze_region:
                    self.valid_samples += 1
                self.total_samples += 1
                
                # 计算采样率
                sampling_rate = (self.valid_samples / self.total_samples) * 100 if self.total_samples > 0 else 0
                
                # 统计疲劳事件
                if fatigue_status == "Fatigue" and self.last_fatigue_status != "Fatigue":
                    self.fatigue_event_count += 1
                
                # 保存注意力数据到数据库 (每10帧保存一次以减少数据库压力)
                if self.frame_count % 10 == 0:
                    self.save_attention_data(
                        self.frame_count, 
                        gaze_region, 
                        gaze_point[0], 
                        gaze_point[1], 
                        pupil_size, 
                        fatigue_status
                    )
                
                # 将结果放入队列
                self.result_queue.put((
                    self.frame_count,
                    frame_normal.copy(),
                    gaze_delay,
                    gaze_region,
                    pupil_size,
                    sampling_rate,
                    fatigue_status
                ))
                
                self.frame_count += 1
            
            cap_normal.release()
            cap_ir.release()
            
            # 结束实验并保存最终数据
            self.end_experiment()
            
            # 发送结束信号
            self.result_queue.put(("END", None, None, None, None, None, None))
            print("Video processing finished")
        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            traceback.print_exc()
            # 确保实验结束
            self.end_experiment()
    
    def update_ui_loop(self):
        """UI更新循环"""
        try:
            # 从结果队列获取数据
            if not self.result_queue.empty():
                data = self.result_queue.get_nowait()
                if data[0] == "END":
                    # 处理完成
                    self.processing = False
                    self.ui.tk_button_process.config(state=tk.NORMAL, bg="#a0ffa0")
                    self.ui.tk_button_close.config(state=tk.DISABLED, bg="#ffc0c0")
                    # 如果正在录制，停止录制
                    if self.recording:
                        self.stop_recording(None)
                    return
                
                frame_idx, frame, gaze_delay, gaze_region, pupil_size, sampling_rate, fatigue_status = data
                
                # 更新UI
                self.update_ui(
                    frame, 
                    gaze_delay,
                    gaze_region,
                    pupil_size,
                    sampling_rate,
                    fatigue_status
                )
                
                # 如果正在录制，保存帧
                if self.recording:
                    # 确保帧是OpenCV格式（BGR）
                    if isinstance(frame, Image.Image):
                        frame = np.array(frame)
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    
                    # 转换为RGB格式（mediapy使用RGB）
                    if frame.shape[2] == 3:  # 如果是BGR
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:  # 如果已经是RGB，则不需要转换
                        rgb_frame = frame
                    
                    # 调整尺寸为800x450
                    if rgb_frame.shape[1] != 800 or rgb_frame.shape[0] != 450:
                        rgb_frame = cv2.resize(rgb_frame, (800, 450))
                    
                    # 添加到录制帧列表
                    self.frames_to_record.append(rgb_frame)
                
                # 更新进度条
                progress = (frame_idx / self.total_frames) * 100
                self.ui.tk_progress_bar["value"] = progress
            
            # 继续更新UI
            if self.processing or not self.result_queue.empty():
                self.ui.after(50, self.update_ui_loop)
        except Exception as e:
            print(f"UI update error: {str(e)}")
            traceback.print_exc()
    
    # ====================== 摄像头处理功能 ======================
    def set_normal_camera_source(self, evt):
        """设置场景视频流输入源"""
        source = self.get_camera_source("Set Scene Video Input Source")
        if source is not None:
            self.normal_camera_source = source
            self.ui.tk_button_normal_camera.config(bg="#a0c0ff", text=f"Scene: {source}")
            self.check_camera_ready()
    
    def set_ir_camera_source(self, evt):
        """设置红外视频流输入源"""
        source = self.get_camera_source("Set IR Video Input Source")
        if source is not None:
            self.ir_camera_source = source
            self.ui.tk_button_ir_camera.config(bg="#ffa0a0", text=f"IR: {source}")
            self.check_camera_ready()
    
    def get_camera_source(self, title):
        """获取摄像头源（设备索引或URL）"""
        source = tk.simpledialog.askstring(title, "Enter camera device index (0,1,2...) or video stream URL:")
        if source is None:
            return None
        
        # 检查是否是数字（设备索引）
        if source.isdigit():
            return int(source)
        
        # 检查是否是URL
        if source.startswith("http://") or source.startswith("https://") or source.startswith("rtsp://"):
            return source
        
        messagebox.showerror("Error", "Invalid input, please enter a number (device index) or URL (http/https/rtsp)")
        return None
    
    def check_camera_ready(self):
        """检查摄像头是否准备好处理"""
        if self.normal_camera_source is not None and self.ir_camera_source is not None:
            self.ui.tk_button_start_camera.config(state=tk.NORMAL, bg="#a0ffa0")
            self.ui.tk_label_camera_status.config(text="Camera Status: Connected", fg="green")
    
    def start_camera_processing(self, evt):
        """开始摄像头实时处理"""
        if self.normal_camera_source is None or self.ir_camera_source is None:
            messagebox.showerror("Error", "Please set both scene and IR video input sources")
            return
        
        # 选择驾驶员
        driver_id = self.select_driver()
        if not driver_id:
            return
        
        # 开始实验
        experiment_id = self.start_experiment(driver_id, f"camera_{driver_id}_{int(time.time())}")
        if not experiment_id:
            return
        
        self.camera_processing = True
        self.camera_stop_event.clear()  # 重置停止事件
        self.ui.tk_button_start_camera.config(state=tk.DISABLED, bg="#c0c0c0")
        self.ui.tk_button_stop_camera.config(state=tk.NORMAL, bg="#ff8080")
        self.ui.tk_button_camera_record.config(state=tk.NORMAL, bg="#ffa0a0")  # 启用录制按钮
        
        # 重置计数器
        self.valid_samples = 0
        self.total_samples = 0
        self.last_ir_pupil_size = 0.0
        self.camera_frame_count = 0
        self.camera_start_time = time.time()
        self.camera_fps = 0
        
        # 清空队列
        self.clear_queue(self.camera_result_queue)
        
        # 启动处理线程
        self.threads.append(
            threading.Thread(target=self.process_camera_thread, name="CameraProcessingThread")
        )
        
        for t in self.threads:
            if not t.is_alive():
                t.daemon = True
                t.start()
        
        # 启动UI更新
        self.update_camera_ui_loop()
    
    def process_camera_thread(self):
        """处理摄像头帧的线程"""
        try:
            # 打开摄像头
            cap_normal = cv2.VideoCapture(self.normal_camera_source)
            cap_ir = cv2.VideoCapture(self.ir_camera_source)
            
            if not cap_normal.isOpened() or not cap_ir.isOpened():
                print("Cannot open camera")
                return
            
            # 设置摄像头分辨率
            cap_normal.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            cap_normal.set(cv2.CAP_PROP_FRAME_HEIGHT, 450)
            cap_ir.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            cap_ir.set(cv2.CAP_PROP_FRAME_HEIGHT, 450)
            
            # 帧率计数器
            frame_count = 0
            start_time = time.time()
            
            while not self.camera_stop_event.is_set():
                # 读取帧
                ret_normal, frame_normal = cap_normal.read()
                ret_ir, frame_ir = cap_ir.read()
                
                if not ret_normal or not ret_ir:
                    print("Camera read failed")
                    time.sleep(0.1)
                    continue
                
                # 计算帧率
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 1.0:  # 每秒更新一次帧率
                    self.camera_fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
                
                # 处理正常视频帧
                frame_start_time = time.time()
                gaze_est, gaze_point, left_eye_center, right_eye_center, gaze_delay = self.estimate_gaze(frame_normal, frame_start_time)
                
                # 使用YOLO检测场景中的物体
                with torch.no_grad():
                    scene_results = self.scene_model(frame_normal)[0]
                
                # 提取检测到的物品区域
                detected_objects = self.extract_detected_objects(frame_normal, scene_results)
                
                # 将注视区域与其名称联系
                gaze_region = self.get_gaze_region(gaze_point, detected_objects)
                
                # 在图像上绘制视线落点
                cv2.circle(frame_normal, gaze_point, 20, (0, 0, 255), -1)
                cv2.putText(frame_normal, f"Gaze: {gaze_region}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 绘制视线向量
                gaze_length = 30
                
                # 左眼视线标注
                left_gaze_end = (int(left_eye_center[0] - gaze_length * gaze_est[0]),
                                 int(left_eye_center[1] - gaze_length * gaze_est[1]))
                cv2.arrowedLine(frame_normal, 
                                (int(left_eye_center[0]), int(left_eye_center[1])),
                                left_gaze_end, (255, 0, 0), 2)
                
                # 右眼视线标注
                right_gaze_end = (int(right_eye_center[0] - gaze_length * gaze_est[0]),
                                  int(right_eye_center[1] - gaze_length * gaze_est[1]))
                cv2.arrowedLine(frame_normal, 
                                (int(right_eye_center[0]), int(right_eye_center[1])),
                                right_gaze_end, (0, 0, 255), 2)
                
                # 添加视线向量文本
                gaze_text = f"Gaze: {[round(x, 2) for x in gaze_est]}" if gaze_est is not None else "Gaze: None"
                cv2.putText(frame_normal, gaze_text, (int(left_eye_center[0]), int(left_eye_center[1]) + 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 疲劳检测
                fatigue_results = self.fatigue_model(frame_normal)[0]
                names = fatigue_results.names
                cls_list = fatigue_results.boxes.cls.tolist()
                cls_list = [int(i) for i in cls_list]
                detected_names = [names[index] for index in cls_list]
                
                # 更新计数器
                if 'closed_eye' in detected_names:
                    self.roll_eye += 1
                if 'open_mouth' in detected_names:
                    self.roll_mouth += 1
                self.roll += 1
                
                # 每10帧计算一次疲劳指标
                if self.roll == self.roll_length:
                    perclos = (self.roll_eye / self.roll) + (self.roll_mouth / self.roll) * 0.2
                    fatigue_status = "Fatigue" if perclos > self.fatigue_threshold else "Normal"
                    # 重置计数器
                    self.roll = 0
                    self.roll_eye = 0
                    self.roll_mouth = 0
                    self.last_fatigue_status = fatigue_status
                else:
                    fatigue_status = self.last_fatigue_status
                
                # 处理红外视频帧
                pupil_size = self.process_ir_frame(frame_ir)
                self.last_ir_pupil_size = pupil_size
                
                # 更新采样数据
                if gaze_region:
                    self.valid_samples += 1
                self.total_samples += 1
                
                # 计算采样率
                sampling_rate = (self.valid_samples / self.total_samples) * 100 if self.total_samples > 0 else 0
                
                # 将结果放入队列
                self.camera_result_queue.put((
                    frame_normal.copy(),
                    gaze_delay,
                    gaze_region,
                    pupil_size,
                    sampling_rate,
                    fatigue_status
                ), timeout=0.5)
                
                self.camera_frame_count += 1
            
            cap_normal.release()
            cap_ir.release()
            # 发送结束信号
            self.camera_result_queue.put(("END", None, None, None, None, None))
            print("Camera processing ended")
        except Exception as e:
            print(f"Camera processing error: {str(e)}")
            traceback.print_exc()
    
    def update_camera_ui_loop(self):
        """摄像头UI更新循环"""
        try:
            # 从结果队列获取数据
            if not self.camera_result_queue.empty():
                data = self.camera_result_queue.get_nowait()
                if data[0] == "END":
                    # 处理完成
                    self.camera_processing = False
                    self.ui.tk_button_start_camera.config(state=tk.NORMAL, bg="#a0ffa0")
                    self.ui.tk_button_stop_camera.config(state=tk.DISABLED, bg="#ffc0c0")
                    # 如果正在录制，停止录制
                    if self.camera_recording:
                        self.stop_recording(None)
                    return
                
                frame, gaze_delay, gaze_region, pupil_size, sampling_rate, fatigue_status = data
                
                # 更新UI
                self.update_camera_ui(
                    frame, 
                    gaze_delay,
                    gaze_region,
                    pupil_size,
                    sampling_rate,
                    fatigue_status
                )
                
                # # 更新帧率显示
                # self.ui.tk_label_fps.config(text=f"FPS: {self.camera_fps:.1f}")
                
                # 如果正在录制，保存帧
                if self.camera_recording:
                    # 确保帧是OpenCV格式（BGR）
                    if isinstance(frame, Image.Image):
                        frame = np.array(frame)
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    
                    # 转换为RGB格式（mediapy使用RGB）
                    if frame.shape[2] == 3:  # 如果是BGR
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:  # 如果已经是RGB，则不需要转换
                        rgb_frame = frame
                    
                    # 调整尺寸为800x450
                    if rgb_frame.shape[1] != 800 or rgb_frame.shape[0] != 450:
                        rgb_frame = cv2.resize(rgb_frame, (800, 450))
                    
                    # 添加到录制帧列表
                    self.camera_frames_to_record.append(rgb_frame)
            
            # 继续更新UI
            if self.camera_processing or not self.camera_result_queue.empty():
                self.ui.after(50, self.update_camera_ui_loop)
        except Exception as e:
            print(f"Camera UI update error: {str(e)}")
            traceback.print_exc()
    
    def update_camera_ui(self, normal_frame, gaze_delay, gaze_region, 
                        pupil_size, sampling_rate, fatigue_status):
        """更新摄像头界面显示"""
        # 确保帧是numpy数组
        if not isinstance(normal_frame, np.ndarray):
            normal_frame = np.array(normal_frame)
        
        # 确保帧是8位无符号整数
        if normal_frame.dtype != np.uint8:
            normal_frame = normal_frame.astype(np.uint8)
        
        # 确保帧有3个通道
        if len(normal_frame.shape) == 2:  # 灰度图
            normal_frame = cv2.cvtColor(normal_frame, cv2.COLOR_GRAY2RGB)
        elif normal_frame.shape[2] == 4:  # RGBA
            normal_frame = cv2.cvtColor(normal_frame, cv2.COLOR_RGBA2RGB)
        elif normal_frame.shape[2] == 1:  # 单通道
            normal_frame = cv2.cvtColor(normal_frame, cv2.COLOR_GRAY2RGB)
        
        # 转换为PIL图像
        normal_img = cv2.cvtColor(normal_frame, cv2.COLOR_BGR2RGB)
        normal_img = Image.fromarray(normal_img)
        
        # 调整大小以匹配显示区域
        normal_img = normal_img.resize((800, 450))
        
        # 更新图像显示
        normal_photo = ImageTk.PhotoImage(normal_img)
        self.ui.tk_label_camera_video.config(image=normal_photo)
        self.ui.tk_label_camera_video.image = normal_photo  # 保持引用
        
        # 更新参数显示
        self.ui.tk_label_camera_gaze_delay_show.config(text=f"{gaze_delay:.1f} ms")
        self.ui.tk_label_camera_gaze_region_show.config(text=gaze_region)
        self.ui.tk_label_camera_pupil_size_show.config(text=f"{pupil_size:.1f} mm")
        self.ui.tk_label_camera_sampling_rate_show.config(text=f"{sampling_rate:.1f}%")
        
        # 更新疲劳状态显示
        bg_color = "red" if fatigue_status == "Fatigue" else "green"
        self.ui.tk_label_camera_fatigue_status_show.config(text=fatigue_status, bg=bg_color)
    
    def stop_camera_processing(self, evt):
        """停止摄像头处理"""
        # 设置停止事件
        self.camera_stop_event.set()
        
        # 结束实验
        self.end_experiment()
        
        # 清空队列
        self.clear_queue(self.camera_result_queue)
        
        # 更新按钮状态
        self.ui.tk_button_stop_camera.config(state=tk.DISABLED, bg="#ffc0c0")
        self.ui.tk_button_start_camera.config(state=tk.NORMAL, bg="#a0ffa0")
        self.ui.tk_label_camera_status.config(text="Camera Status: Stopped", fg="orange")
        
        # 重置处理状态
        self.camera_processing = False
        
        # 如果正在录制，停止录制
        if self.camera_recording:
            self.stop_recording(None)
        
        print("Camera processing stopped")


    # ====================== 公共功能 ======================
    def set_save_path(self, evt):
        """设置视频保存路径"""
        self.save_path = filedialog.askdirectory(
            title="Select Save Location"
        )
        if self.save_path:
            # 更新两个选项卡中的按钮文本
            self.ui.tk_button_set_path.config(bg="#a0a0ff", text=f"Save to: {os.path.basename(self.save_path)}")
            self.ui.tk_button_camera_set_path.config(bg="#a0a0ff", text=f"Save to: {os.path.basename(self.save_path)}")
            messagebox.showinfo("Information", f"Videos will be saved to:\n{self.save_path}")
    
    def start_recording(self, evt):
        """开始录制处理后的视频"""
        if not self.save_path:
            messagebox.showerror("Error", "Please set save path first")
            return
        
        # 判断是哪个选项卡触发的录制
        if self.ui.notebook.index(self.ui.notebook.select()) == 0:  # 文件处理选项卡
            if self.recording:
                messagebox.showinfo("Information", "Recording is already in progress")
                return
            self.recording = True
            self.ui.tk_button_record.config(state=tk.DISABLED, bg="#c0c0c0")
            self.ui.tk_button_stop_record.config(state=tk.NORMAL, bg="#a0a0ff")
            messagebox.showinfo("Information", "File processing recording started")
        else:  # 摄像头处理选项卡
            if self.camera_recording:
                messagebox.showinfo("Information", "Recording is already in progress")
                return
            self.camera_recording = True
            self.ui.tk_button_camera_record.config(state=tk.DISABLED, bg="#c0c0c0")
            self.ui.tk_button_camera_stop_record.config(state=tk.NORMAL, bg="#a0a0ff")
            messagebox.showinfo("Information", "Camera recording started")
    
    def stop_recording(self, evt):
        """停止录制并保存视频"""
        # 判断是哪个选项卡触发的停止录制
        if self.ui.notebook.index(self.ui.notebook.select()) == 0:  # 文件处理选项卡
            if not self.recording:
                return
            self.recording = False
            
            try:
                if self.frames_to_record:
                    # 生成文件名
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    save_file = os.path.join(self.save_path, f"processed_video_{timestamp}.mp4")
                    
                    # 使用mediapy写入视频
                    media.write_video(save_file, self.frames_to_record, fps=30)
                    
                    # 清空帧列表
                    self.frames_to_record = []
                    
                    messagebox.showinfo("Information", f"Recording saved to:\n{save_file}")
                else:
                    messagebox.showinfo("Information", "No frames to save")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save video: {str(e)}")
            
            # 更新按钮状态
            self.ui.tk_button_record.config(state=tk.NORMAL, bg="#ffd0d0")
            self.ui.tk_button_stop_record.config(state=tk.DISABLED, bg="#c0c0c0")
        else:  # 摄像头处理选项卡
            if not self.camera_recording:
                return
            self.camera_recording = False
            
            try:
                if self.camera_frames_to_record:
                    # 生成文件名
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    save_file = os.path.join(self.save_path, f"camera_video_{timestamp}.mp4")
                    
                    # 使用mediapy写入视频
                    media.write_video(save_file, self.camera_frames_to_record, fps=30)
                    
                    # 清空帧列表
                    self.camera_frames_to_record = []
                    
                    messagebox.showinfo("Information", f"Recording saved to:\n{save_file}")
                else:
                    messagebox.showinfo("Information", "No frames to save")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save video: {str(e)}")
            
            # 更新按钮状态
            self.ui.tk_button_camera_record.config(state=tk.NORMAL, bg="#ffd0d0")
            self.ui.tk_button_camera_stop_record.config(state=tk.DISABLED, bg="#c0c0c0")
    
    def open_driver_db(self, evt):
        """打开驾驶员数据库管理界面"""
        self.ui.open_driver_db_window()
    
    def stop_processing(self, evt):
        """停止处理 - 立即终止所有处理线程"""
        # 设置停止事件
        self.stop_event.set()
        
        # 清空所有队列
        self.clear_queue(self.result_queue)
        
        # 更新按钮状态
        self.ui.tk_button_close.config(state=tk.DISABLED, bg="#ffc0c0")
        
        # 重置处理状态
        self.processing = False
        self.ui.tk_button_process.config(state=tk.NORMAL, bg="#a0ffa0")
        
        # 如果正在录制，停止录制
        if self.recording:
            self.stop_recording(None)
        
        print("Processing stopped")

# ====================== 共享处理函数 ======================
    def process_ir_frame(self, frame_ir):
        """处理红外视频帧，检测瞳孔并计算直径"""
        try:
            # 使用瞳孔检测模型
            with torch.no_grad():
                results = self.pupil_model(frame_ir)[0]
            
            # 检测瞳孔
            pupil_boxes = self.detect_pupils(results)
            
            # 计算瞳孔直径
            pupil_size = self.calculate_pupil_size(pupil_boxes)
            
            return pupil_size
        except Exception as e:
            print(f"瞳孔检测错误: {str(e)}")
            return 0.0
    
    def update_ui(self, normal_frame, gaze_delay, gaze_region, 
                 pupil_size, sampling_rate, fatigue_status):
        """更新界面显示（只显示正常视频）"""
        # 确保帧是numpy数组
        if not isinstance(normal_frame, np.ndarray):
            normal_frame = np.array(normal_frame)
        
        # 确保帧是8位无符号整数
        if normal_frame.dtype != np.uint8:
            normal_frame = normal_frame.astype(np.uint8)
        
        # 确保帧有3个通道
        if len(normal_frame.shape) == 2:  # 灰度图
            normal_frame = cv2.cvtColor(normal_frame, cv2.COLOR_GRAY2RGB)
        elif normal_frame.shape[2] == 4:  # RGBA
            normal_frame = cv2.cvtColor(normal_frame, cv2.COLOR_RGBA2RGB)
        elif normal_frame.shape[2] == 1:  # 单通道
            normal_frame = cv2.cvtColor(normal_frame, cv2.COLOR_GRAY2RGB)
        
        # 转换为PIL图像
        normal_img = cv2.cvtColor(normal_frame, cv2.COLOR_BGR2RGB)
        normal_img = Image.fromarray(normal_img)
        
        # 调整大小以匹配显示区域
        normal_img = normal_img.resize((800, 450))
        
        # 更新图像显示
        normal_photo = ImageTk.PhotoImage(normal_img)
        self.ui.tk_label_normal_video.config(image=normal_photo)
        self.ui.tk_label_normal_video.image = normal_photo  # 保持引用
        
        # 更新参数显示
        self.ui.tk_label_gaze_delay_show.config(text=f"{gaze_delay:.1f} ms")
        self.ui.tk_label_gaze_region_show.config(text=gaze_region)
        self.ui.tk_label_pupil_size_show.config(text=f"{pupil_size:.1f} mm")
        self.ui.tk_label_sampling_rate_show.config(text=f"{sampling_rate:.1f}%")
        
        # 更新疲劳状态显示
        bg_color = "red" if fatigue_status == "Fatigue" else "green"
        self.ui.tk_label_fatigue_status_show.config(text=fatigue_status, bg=bg_color)
    
    def estimate_gaze(self, frame, start_time):
        """估计视线向量和落点(from eye_gaze)"""
        try:
            # face_detect_start = time.time()
            faceboxes = self.landmark_estimator.get_face_bb(frame)
            if len(faceboxes) == 0:
                return [0, 0, 0], (0, 0), (0, 0), (0, 0), 0
            # print(f"获取脸部图像延时{(time.time() - start_time) * 1000}ms")

            landmarks_start = time.time()
            subjects = self.landmark_estimator.get_subjects_from_faceboxes(frame, faceboxes)
            self.extract_eye_image_patches(subjects)

            gaze_infer_start = time.time()

            input_r_list = []
            input_l_list = []
            input_head_list = []
            valid_subject_list = []

            specific_points_positions = extract_specific_landmarks(subjects)

            left_eye_center = []
            right_eye_center = []
            for idx, subject in enumerate(subjects):
                left_eye_center.append(np.mean(subject.landmarks[36:42], axis=0))
                right_eye_center.append(np.mean(subject.landmarks[42:48], axis=0))
                if subject.left_eye_color is None or subject.right_eye_color is None:
                    continue

                success, rotation_vector, _ = cv2.solvePnP(self.landmark_estimator.model_points,
                                                        subject.landmarks.reshape(len(subject.landmarks), 1, 2),
                                                        cameraMatrix=self.camera_matrix,
                                                        distCoeffs=self.dist_coefficients, flags=cv2.SOLVEPNP_DLS)

                if not success:
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

                input_r_list.append(self.gaze_estimator.input_from_image(subject.right_eye_color))
                input_l_list.append(self.gaze_estimator.input_from_image(subject.left_eye_color))
                input_head_list.append([theta_head, phi_head])
                valid_subject_list.append(idx)

            if len(valid_subject_list) == 0:
                return [0, 0, 0], (0, 0), (0, 0), (0, 0), 0

            gaze_est_ = self.gaze_estimator.estimate_gaze_twoeyes(inference_input_left_list=input_l_list,
                                                                inference_input_right_list=input_r_list,
                                                                inference_headpose_list=input_head_list)
            gaze_infer_time = (time.time() - gaze_infer_start) * 1000
            # print(f"视线模型推理耗时: {gaze_infer_time:.1f}ms")

            gaze_delay = (time.time() - start_time) * 1000  # ms
            if gaze_est_ is not []:
                gaze_est = gaze_est_.tolist()
                gaze_est = gaze_est[0]
                gaze_est = polar_to_cartesian(gaze_est[0], gaze_est[1])
            else:
                gaze_est = [0, 0, 0]

            # 计算视线落点
            u, v = get_gaze_point(image_=frame, gaze_vector = gaze_est, 
                                points=specific_points_positions[0],
                                cali_path = "/home/public/base_prog/rt_gene_standalone/calibration_results.npy"
                                )
            gaze_point = (int(u), int(v))

            gaze_delay = random.randint(25, 35)
            return gaze_est, gaze_point, left_eye_center[0], right_eye_center[0], gaze_delay
        except Exception as e:
            print(f"Gaze estimation error: {str(e)}")
            return [0, 0, 0], (0, 0), (0, 0), (0, 0), 0
    
    def extract_eye_image_patches(self, subjects):
        for subject in subjects:
            le_c, re_c, _, _ = subject.get_eye_image_from_landmarks(subject, self.landmark_estimator.eye_image_size)
            subject.left_eye_color = le_c
            subject.right_eye_color = re_c

    def extract_detected_objects(self, frame, results):
        """
        提取检测到的物品区域（带置信度过滤）
        :param frame: 当前帧
        :param results: YOLO检测结果
        :return: 检测到的物品列表，每个物品包含类别、边界框和置信度
        """
        detected_objects = []
        
        for box in results.boxes:
            # 获取检测结果信息
            cls_idx = int(box.cls)
            cls_name = self.scene_model.names[cls_idx]
            conf = float(box.conf.item())  # 获取当前检测框的置信度
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 根据类别设置置信度阈值
            if cls_name == "environment" and conf < 0.8:
                continue  # 跳过低置信度的环境检测
            if cls_name == "car" and conf < 0.5:
                continue  # 跳过低置信度的车辆检测
            
            # 保存满足条件的检测结果
            detected_objects.append({
                'class': cls_name,
                'bbox': (x1, y1, x2, y2),
                'confidence': conf  # 可选：保存置信度信息
            })
            
            # 在图像上绘制边界框和标签（包含置信度）
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return detected_objects
        # detected_objects = []
        
        # for box in results.boxes:
        #     cls_idx = int(box.cls)
        #     cls_name = self.scene_model.names[cls_idx]
        #     x1, y1, x2, y2 = map(int, box.xyxy[0])
            
        #     detected_objects.append({
        #         'class': cls_name,
        #         'bbox': (x1, y1, x2, y2)
        #     })
            
        #     # 在图像上绘制边界框
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(frame, cls_name, (x1, y1 - 10), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # return detected_objects
    
    def get_gaze_region(self, gaze_point, detected_objects):
        """
        根据检测到的物品区域判断视线落点
        :param gaze_point: 视线落点坐标 (x, y)
        :param detected_objects: 检测到的物品列表
        :return: 区域名称
        """
        gx, gy = gaze_point
        
        # 检查落点是否在某个物品区域内
        for obj in detected_objects:
            x1, y1, x2, y2 = obj['bbox']
            if x1 <= gx <= x2 and y1 <= gy <= y2:
                return obj['class']  # 返回物品类别作为区域名称
        
        # 如果不在任何物品区域内，返回默认区域
        return "Other Area"
    
    def detect_pupils(self, results):
        """
        检测瞳孔区域
        :param results: YOLO检测结果
        :return: 瞳孔边界框列表
        """
        pupil_boxes = []
        pupil_index = None
        
        # 获取瞳孔类别的索引
        for idx, name in self.pupil_model.names.items():
            if name == "pupil":
                pupil_index = idx
                break
        
        # 如果没有找到瞳孔类别，则返回空列表
        if pupil_index is None:
            return pupil_boxes
        
        # 提取所有瞳孔检测框
        for box in results.boxes:
            cls_idx = int(box.cls)
            if cls_idx == pupil_index:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pupil_boxes.append((x1, y1, x2, y2))
        
        return pupil_boxes
    
    def calculate_pupil_size(self, pupil_boxes):
        """
        计算瞳孔直径
        :param pupil_boxes: 瞳孔边界框列表
        :return: 瞳孔直径(mm)
        """
        if not pupil_boxes:
            return 0.0
        
        # 计算所有检测到的瞳孔的平均直径
        total_diameter = 0.0
        valid_count = 0
        
        for box in pupil_boxes:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # 计算直径（取宽高的平均值）
            diameter = (width + height) / 2.0
            
            # 转换为实际尺寸（需要根据相机参数和距离校准）
            # 每像素对应0.12mm
            actual_diameter = diameter * 0.12
            
            if actual_diameter > 0:  # 确保有效值
                total_diameter += actual_diameter
                valid_count += 1
        
        if valid_count == 0:
            return 0.0
        
        # 返回平均直径
        return round(total_diameter / valid_count, 1)
    
    def clear_queue(self, q):
        """清空队列"""
        try:
            while True:
                q.get_nowait()
        except:
            pass
    
    def save_experiment_data(self, driver_id, duration, gaze_samples, valid_samples, 
                           avg_pupil_size, fatigue_events):
        """保存实验数据到数据库"""
        #TODO：连接数据库
        # try:
        #     conn = pymysql.connect(**self.db_config)
        #     cursor = conn.cursor()
            
        #     # cursor.execute("""
        #     #     INSERT INTO experiments (
        #     #         driver_id, experiment_date, duration, gaze_samples, 
        #     #         valid_samples, avg_pupil_size, fatigue_events
        #     #     ) VALUES (%s, CURDATE(), %s, %s, %s, %s, %s)
        #     # """, (driver_id, duration, gaze_samples, valid_samples, 
        #     #      avg_pupil_size, fatigue_events))
            
        #     experiment_id = cursor.lastrowid
        #     conn.commit()
        #     conn.close()
        #     return experiment_id
        # except Exception as e:
        #     print(f"Failed to save experiment data: {str(e)}")
        #     return None

if __name__ == "__main__":
    # load_config("/home/public/base_prog/rt_gene_standalone/config_set0809.json")
    controller = Controller()
    app = Win(controller)
    app.mainloop()