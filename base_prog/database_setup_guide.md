# 驾驶员注意力检测系统 - 数据库配置说明

## 1. 数据库设置

### MySQL安装和配置
1. 确保MySQL服务已启动
2. 创建数据库用户（如果需要）
3. 修改数据库连接参数

### 配置文件位置
- 主程序: `gui_english.py` 中的 `db_config` 字典
- 设置脚本: `setup_database.py` 中的 `db_config` 字典

### 需要修改的参数
```python
db_config_db = {
        "host": "localhost",
        "user": "root",
        "password": "",
        "database": "driver_attention_db",
        "unix_socket": "/home/public/mysql/mysql.sock",
        "port": 3307,
        "charset": "utf8mb4"
    }
```

## 2. 数据库初始化

### 运行设置脚本
```bash
cd /home/public/base_prog
python setup_database.py
```

### 手动创建数据库
如果自动设置失败，可以手动执行以下SQL：

```sql
-- 创建数据库
CREATE DATABASE driver_attention_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE driver_attention_db;

-- 创建驾驶员表
CREATE TABLE drivers (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL COMMENT '驾驶员姓名',
    employee_id VARCHAR(20) NOT NULL UNIQUE COMMENT '员工ID',
    license_type VARCHAR(10) NOT NULL COMMENT '驾照类型',
    phone VARCHAR(20) COMMENT '电话号码',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间'
);

-- 创建实验数据表
CREATE TABLE experiments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    driver_id INT NOT NULL COMMENT '驾驶员ID',
    experiment_date DATE NOT NULL COMMENT '实验日期',
    start_time TIMESTAMP NOT NULL COMMENT '开始时间',
    end_time TIMESTAMP NULL COMMENT '结束时间',
    duration INT DEFAULT 0 COMMENT '持续时间(秒)',
    total_frames INT DEFAULT 0 COMMENT '总帧数',
    gaze_samples INT DEFAULT 0 COMMENT '注视样本数',
    valid_samples INT DEFAULT 0 COMMENT '有效样本数',
    avg_pupil_size FLOAT DEFAULT 0 COMMENT '平均瞳孔大小',
    fatigue_events INT DEFAULT 0 COMMENT '疲劳事件数',
    video_path VARCHAR(500) COMMENT '视频路径',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    FOREIGN KEY (driver_id) REFERENCES drivers(id) ON DELETE CASCADE
);

-- 创建注意力分配数据表
CREATE TABLE attention_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id INT NOT NULL COMMENT '实验ID',
    frame_number INT NOT NULL COMMENT '帧号',
    timestamp TIMESTAMP NOT NULL COMMENT '时间戳',
    gaze_region VARCHAR(50) NOT NULL COMMENT '注视区域',
    gaze_x FLOAT COMMENT '注视点X坐标',
    gaze_y FLOAT COMMENT '注视点Y坐标',
    pupil_size FLOAT COMMENT '瞳孔大小',
    fatigue_status VARCHAR(20) COMMENT '疲劳状态',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
);
```

## 3. 功能说明

### 驾驶员管理
- 添加新驾驶员
- 编辑驾驶员信息
- 删除驾驶员（会同时删除相关实验数据）
- 搜索和筛选驾驶员
- 导出驾驶员数据到CSV
- 统计信息查看

### 实验数据管理
- 开始视频处理或摄像头处理时自动选择驾驶员
- 实时保存注意力数据
- 查看驾驶员的历史实验记录
- 查看实验详细数据
- 自动计算统计指标

### 数据结构
- drivers: 驾驶员基本信息
- experiments: 实验概要数据
- attention_data: 详细的帧级注意力数据

## 4. 使用流程

1. 配置数据库连接参数
2. 运行 `setup_database.py` 初始化数据库
3. 启动主程序 `gui_english.py`
4. 点击 "Driver Database" 管理驾驶员信息
5. 处理视频时选择对应的驾驶员
6. 系统自动保存实验数据
7. 通过数据库管理界面查看和分析数据

## 5. 故障排除

### 连接失败
- 检查MySQL服务是否启动
- 确认用户名和密码正确
- 检查防火墙设置
- 确认数据库存在

### 权限问题
- 确保数据库用户有足够权限
- 检查表的创建权限
- 确认外键约束设置正确

### 性能优化
- 注意力数据量较大，建议定期清理旧数据
- 可以调整数据保存频率（目前每10帧保存一次）
- 考虑添加索引优化查询性能
