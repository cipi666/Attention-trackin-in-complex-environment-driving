#!/usr/bin/env python3
"""
数据库设置脚本
用于初始化驾驶员注意力检测系统的MySQL数据库
"""

import pymysql
import sys

def create_database():
    """创建数据库和表"""
    
    # 使用与demo.py相同的数据库连接配置
    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "",
        "unix_socket": "/home/public/mysql/mysql.sock",
        "port": 3307,
        "charset": "utf8mb4"
    }
    
    # 连接到具体数据库的配置
    db_config_db = {
        "host": "localhost",
        "user": "root",
        "password": "",
        "database": "driver_attention_db",
        "unix_socket": "/home/public/mysql/mysql.sock",
        "port": 3307,
        "charset": "utf8mb4"
    }
    
    try:
        print("连接MySQL服务器...")
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        
        # 创建数据库
        print("创建数据库 driver_attention_db...")
        cursor.execute("CREATE DATABASE IF NOT EXISTS driver_attention_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        cursor.close()
        conn.close()
        
        # 重新连接到新创建的数据库
        print("连接到 driver_attention_db 数据库...")
        conn = pymysql.connect(**db_config_db)
        cursor = conn.cursor()
        
        # 创建驾驶员表
        print("创建驾驶员表...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drivers (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL COMMENT 'Driver name',
                employee_id VARCHAR(20) NOT NULL UNIQUE COMMENT 'Employee ID',
                license_type VARCHAR(10) NOT NULL COMMENT 'License type',
                phone VARCHAR(20) COMMENT 'Phone number',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Created time'
            ) ENGINE=InnoDB COMMENT='Driver information table'
        """)
        
        # 创建实验数据表
        print("创建实验数据表...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INT AUTO_INCREMENT PRIMARY KEY,
                driver_id INT NOT NULL COMMENT 'Driver ID',
                experiment_date DATE NOT NULL COMMENT 'Experiment date',
                start_time TIMESTAMP NOT NULL COMMENT 'Start time',
                end_time TIMESTAMP NULL COMMENT 'End time',
                duration INT DEFAULT 0 COMMENT 'Duration in seconds',
                total_frames INT DEFAULT 0 COMMENT 'Total frames',
                gaze_samples INT DEFAULT 0 COMMENT 'Gaze samples',
                valid_samples INT DEFAULT 0 COMMENT 'Valid samples',
                avg_pupil_size FLOAT DEFAULT 0 COMMENT 'Average pupil size',
                fatigue_events INT DEFAULT 0 COMMENT 'Fatigue events',
                video_path VARCHAR(500) COMMENT 'Video path',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Created time',
                FOREIGN KEY (driver_id) REFERENCES drivers(id) ON DELETE CASCADE,
                INDEX idx_driver_date (driver_id, experiment_date),
                INDEX idx_start_time (start_time)
            ) ENGINE=InnoDB COMMENT='Experiment data table'
        """)
        
        # 创建注意力分配数据表
        print("创建注意力数据表...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attention_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                experiment_id INT NOT NULL COMMENT 'Experiment ID',
                frame_number INT NOT NULL COMMENT 'Frame number',
                timestamp TIMESTAMP NOT NULL COMMENT 'Timestamp',
                gaze_region VARCHAR(50) NOT NULL COMMENT 'Gaze region',
                gaze_x FLOAT COMMENT 'Gaze point X coordinate',
                gaze_y FLOAT COMMENT 'Gaze point Y coordinate',
                pupil_size FLOAT COMMENT 'Pupil size',
                fatigue_status VARCHAR(20) COMMENT 'Fatigue status',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT 'Created time',
                FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
                INDEX idx_experiment_frame (experiment_id, frame_number),
                INDEX idx_timestamp (timestamp)
            ) ENGINE=InnoDB COMMENT='Attention data table'
        """)
        
        # 插入示例数据
        print("插入示例数据...")
        cursor.execute("""
            INSERT IGNORE INTO drivers (name, employee_id, license_type, phone) VALUES
            ('Zhang San', 'EMP001', 'C1', '13800138001'),
            ('Li Si', 'EMP002', 'B2', '13800138002'),
            ('Wang Wu', 'EMP003', 'A1', '13800138003')
        """)
        
        conn.commit()
        print("数据库初始化完成！")
        
        # 显示创建的表
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print(f"创建的表: {[table[0] for table in tables]}")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"数据库初始化失败: {str(e)}")
        sys.exit(1)

def test_connection():
    """测试数据库连接"""
    
    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "",
        "database": "labdb",
        "unix_socket": "/home/public/mysql/mysql.sock",
        "port": 3307,
        "charset": "utf8mb4"
    }
    
    try:
        print("测试MySQL服务器连接...")
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES")
        databases = cursor.fetchall()
        print(f"可用数据库: {[db[0] for db in databases]}")
        cursor.close()
        conn.close()
        print("MySQL服务器连接成功！")
        
        # 检查driver_attention_db是否存在
        if ('driver_attention_db',) in databases:
            # 连接到driver_attention_db
            db_config_db = {
                "host": "localhost",
                "user": "root",
                "password": "",
                "database": "driver_attention_db",
                "unix_socket": "/home/public/mysql/mysql.sock",
                "port": 3307,
                "charset": "utf8mb4"
            }
            
            conn = pymysql.connect(**db_config_db)
            cursor = conn.cursor()
            
            # 查询驾驶员数量
            cursor.execute("SELECT COUNT(*) FROM drivers")
            driver_count = cursor.fetchone()[0]
            print(f"当前驾驶员数量: {driver_count}")
            
            # 查询实验数量
            cursor.execute("SELECT COUNT(*) FROM experiments")
            exp_count = cursor.fetchone()[0]
            print(f"当前实验数量: {exp_count}")
            
            cursor.close()
            conn.close()
            print("driver_attention_db 数据库连接测试成功！")
        else:
            print("driver_attention_db 数据库不存在，请先运行初始化")
        
    except Exception as e:
        print(f"数据库连接测试失败: {str(e)}")
        print("请检查以下项目:")
        print("1. MySQL服务是否正在运行")
        print("2. MySQL socket文件是否存在: /home/public/mysql/mysql.sock")
        print("3. 如果MySQL以skip-grant-tables模式运行，请重启MySQL服务")

if __name__ == "__main__":
    print("=== 驾驶员注意力检测系统数据库设置 ===")
    print("注意: 请先修改脚本中的数据库连接参数")
    print()
    
    choice = input("选择操作:\n1. 初始化数据库\n2. 测试连接\n请输入(1或2): ")
    
    if choice == "1":
        create_database()
    elif choice == "2":
        test_connection()
    else:
        print("无效选择")
