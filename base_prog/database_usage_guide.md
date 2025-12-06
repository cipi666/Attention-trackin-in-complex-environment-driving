# 驾驶员数据库管理功能使用说明

## 概述

我已经成功为你的驾驶员注意力检测系统实现了完整的数据库管理功能。现在系统可以：

1. **驾驶员信息管理**：增删改查、导出、统计
2. **实验数据自动保存**：视频处理和摄像头处理时自动选择驾驶员并保存数据
3. **实验数据查看**：查看历史实验记录和详细数据

## 修改内容总结

### 1. 数据库功能完善
- ✅ 取消注释了所有数据库相关代码
- ✅ 完善了驾驶员信息的增删改查功能
- ✅ 添加了数据验证和错误处理
- ✅ 实现了统计功能和数据导出

### 2. 实验数据自动保存
- ✅ 添加了驾驶员选择对话框
- ✅ 视频处理开始时自动创建实验记录
- ✅ 处理过程中每10帧保存一次注意力数据
- ✅ 处理结束时自动更新实验统计数据

### 3. 数据查看功能
- ✅ 添加了"View Experiments"按钮
- ✅ 可以查看每个驾驶员的历史实验
- ✅ 可以查看实验的详细数据

### 4. 辅助工具
- ✅ 创建了数据库初始化脚本 `setup_database.py`
- ✅ 提供了详细的配置说明文档

## 使用步骤

### 第一步：配置数据库
1. 修改 `gui_english.py` 中的数据库连接参数：
```python
self.db_config_db = {
        "host": "localhost",
        "user": "root",
        "password": "",
        "database": "driver_attention_db",
        "unix_socket": "/home/public/mysql/mysql.sock",
        "port": 3307,
        "charset": "utf8mb4"
}
```

2. 同样修改 `setup_database.py` 中的连接参数

### 第二步：初始化数据库
```bash
cd /home/public/base_prog
python setup_database.py
```
选择"1"来初始化数据库，或选择"2"来测试连接。

### 第三步：使用系统
1. 启动主程序：`python gui_english.py`
2. 点击 "Driver Database" 按钮管理驾驶员信息
3. 开始视频处理时会自动弹出驾驶员选择对话框
4. 系统会自动保存实验数据

## 功能详解

### 驾驶员管理界面
- **搜索功能**：支持按ID、姓名、驾照类型搜索
- **Add Driver**：添加新驾驶员
- **Edit Driver**：编辑选中的驾驶员信息
- **Delete Driver**：删除驾驶员（会同时删除相关实验数据）
- **View Experiments**：查看选中驾驶员的实验记录
- **Export Data**：导出驾驶员列表到CSV文件
- **Statistics**：显示统计信息（总数、类型分布、最近添加等）

### 实验数据自动保存
- 开始处理视频或摄像头时，会弹出驾驶员选择对话框
- 选择驾驶员后自动创建实验记录
- 处理过程中每10帧保存一次注意力数据（可调整频率）
- 保存的数据包括：
  - 帧号
  - 注视区域
  - 注视点坐标
  - 瞳孔大小
  - 疲劳状态
- 处理结束时自动更新实验统计

### 实验数据查看
- 在驾驶员管理界面选择驾驶员后点击"View Experiments"
- 显示该驾驶员的所有实验记录
- 点击"View Details"查看具体实验的详细数据

## 数据库结构

### drivers 表（驾驶员信息）
- id：主键
- name：姓名
- employee_id：员工ID（唯一）
- license_type：驾照类型
- phone：电话
- created_at：创建时间

### experiments 表（实验概要）
- id：主键
- driver_id：驾驶员ID
- experiment_date：实验日期
- start_time/end_time：开始/结束时间
- duration：持续时间（秒）
- total_frames：总帧数
- gaze_samples/valid_samples：总样本数/有效样本数
- avg_pupil_size：平均瞳孔大小
- fatigue_events：疲劳事件数
- video_path：视频路径

### attention_data 表（详细注意力数据）
- id：主键
- experiment_id：实验ID
- frame_number：帧号
- timestamp：时间戳
- gaze_region：注视区域
- gaze_x/gaze_y：注视点坐标
- pupil_size：瞳孔大小
- fatigue_status：疲劳状态

## 注意事项

1. **数据库权限**：确保MySQL用户有足够的权限创建数据库和表
2. **数据量**：注意力数据可能会很大，建议定期清理旧数据
3. **备份**：重要数据请定期备份
4. **性能**：如果数据量很大，可以调整保存频率（修改 `self.frame_count % 10` 中的10）

## 故障排除

### 常见错误
1. **连接失败**：检查MySQL服务、用户名密码、防火墙
2. **权限错误**：确保用户有创建数据库和表的权限
3. **编码问题**：确保使用utf8mb4字符集

### 调试方法
- 查看控制台输出的错误信息
- 使用 `setup_database.py` 测试数据库连接
- 检查MySQL日志

## 后续扩展

可以考虑添加的功能：
1. 数据分析图表
2. 实验报告生成
3. 数据对比分析
4. 自动数据清理
5. 数据导入导出功能

现在你的系统已经具备了完整的数据库功能，可以进行驾驶员管理和实验数据的自动保存！
