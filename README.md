# <center><span style="font-family: SimHei, 黑体;">用户使用手册</span></center>

## <center><span style="font-family: SimHei, 黑体;">1.系统核心功能总览</span></center>

### 1.1.基于场景信息的注视对象检测功能
<span style="font-family: SimSun;">系统通过</span><span style="font-family: 'Times New Roman';">YOLOv8</span><span style="font-family: SimSun;">模型对驾驶员第一视角场景进行实时检测，可精准识别并划分出方向盘、仪表盘、车窗、后视镜、反光镜、雨刮器、环境、道路、路面汽车、车载空调等关键区域。同时，系统整合了开源视线追踪项目<span style="font-family: 'Times New Roman';">RT-GENE</span>，实现系统双目视线方向计算采样率为</span><span style="font-family: 'Times New Roman';">60Hz</span><span style="font-family: SimSun;">，通过获取视线向量，结合眼球特征点、相机参数及物理模型，利用几何投影模型实现从视线向量到视线落点的精准计算，实现基于场景信息的注视对象检测。</span>

### 1.2.驾驶信息统计数据库
<span style="font-family: SimSun;">系统采用</span><span style="font-family: 'Times New Roman';">MySQL</span><span style="font-family: SimSun;">关系型数据库作为核心存储引擎，构建了完整的驾驶员注意力检测数据管理体系。数据架构分为三层：驾驶员基础信息层用于维护驾驶员档案，包含个人信息、驾照类型及联系方式等；实验会话管理层可自动化处理实验全生命周期，从会话创建、进程监控到数据汇总，记录时间戳、持续时长、处理帧数等实验元数据；细粒度行为数据层实现帧级注意力数据的实时采集与存储，精确记录注视点坐标、区域分类、瞳孔尺寸变化及疲劳状态判断结果。</span>

<span style="font-family: SimSun;">系统具备完善的数据管理与分析能力，支持完整的</span><span style="font-family: 'Times New Roman';">CRUD</span><span style="font-family: SimSun;">操作及多维度检索、统计分析，可按驾驶员、时间范围等条件进行数据查询与可视化展示。数据库采用优化的索引策略与分批处理机制，通过可配置采样频率（默认每</span><span style="font-family: 'Times New Roman';">10</span><span style="font-family: SimSun;">帧记录一次）平衡数据精度与系统性能。技术上通过外键约束和事务机制确保数据一致性，采用模块化设计支持功能扩展，并提供图形化管理界面与自动化数据处理流程，同时集成错误处理、连接诊断及数据备份等可靠性保障机制。</span>

<span style="font-family: SimSun;">该数据库无缝集成视频文件处理与实时摄像头检测功能，为驾驶员注意力研究提供标准化数据管理方案，支持从实验设计、数据采集到结果分析的完整流程，可有效支撑驾驶行为分析、疲劳检测评估及安全性评估等应用场景。</span>

### 1.3.视觉注意力分配与视觉注视行为分析功能
<span style="font-family: SimSun;">系统具备全面的视觉注意力分配与注视行为分析能力。通过整合优化的视线追踪技术，结合眼球特征点（如眼角坐标）、视线方向向量、相机参数（焦距、位置偏移）及物理模型（像素压缩比例、相机高度），可精准计算视线在物理空间的落点坐标。</span>

<span style="font-family: SimSun;">为解决个体差异导致的误差，系统配备了自适应标定系统：通过交互界面引导用户完成标定流程，记录</span><span style="font-family: 'Times New Roman';">17</span><span style="font-family: SimSun;">个关键点的注视数据（其中训练集用于模型训练，后续轨迹作为测试集），并提供多种代表点计算方法（中位数法、核密度估计、加权平均法）和映射模型训练方法（薄板样条插值、多项式回归），默认采用核密度估计结合薄板样条插值的组合方案，将原始视线落点校正至真实场景坐标系，显著提升不同个体、不同设备下的检测精度。</span>

<span style="font-family: SimSun;">基于校正后的视线落点数据，系统可生成丰富的分析结果与可视化内容，包括：视线落点累计图（直观展示注意力集中区域）、逐帧视线数据记录表（便于精细化分析）、逐帧分割并标注视线向量与落点的图像集，以及叠加视线轨迹的可视化视频。这些成果可有效支撑注意力分配的量化评估，为驾驶行为分析提供全面数据支持。同时，系统采用轻量化几何模型，计算耗时低于</span><span style="font-family: 'Times New Roman';">30ms</span><span style="font-family: SimSun;">，满足实时性要求，并通过闪烁检测算法自动定位初始标定帧，减少人工干预，提升鲁棒性。</span>

### 1.4.落点场景回放和录制功能
<span style="font-family: SimSun;">系统支持落点场景的实时录制与回放，便于对驾驶过程中的视线行为进行回溯分析。在录制过程中，通过结合标定系统的交互流程，可生成包含标定轨迹及对应面部状态的视频文件，完整记录标定过程中的视线变化与面部特征。</span>

<span style="font-family: SimSun;">对录制的视频进行处理后，系统将输出一系列结构化结果与可视化内容，包括：存储代表点与标定点映射关系的标定结果数据、展示各标定点映射误差及分布的误差图、对比注视代表点与理论点的关系图（直观分析映射合理性）、标定时段内原始眼动点位置的散点图，以及基于时间的标定段划分图（展示各标定点的开始与结束帧）。</span>

<span style="font-family: SimSun;">这些功能不仅实现了驾驶场景中叠加视线轨迹的回看功能，直观展示视线追踪的稳定性与准确性，还通过结构化存储标定结果与原始数据，为后续数据追溯、模型优化及多维度分析提供支持，同时兼容数据库集成，便于数据的统一管理与查询。</span>

## <center><span style="font-family: SimHei, 黑体;">2.系统运行环境说明与搭建</span></center>
<span style="font-family: 'Times New Roman';">Python</span><span style="font-family: SimSun;">版本需为</span><span style="font-family: 'Times New Roman';">3.9</span><span style="font-family: SimSun;">，建议使用</span><span style="font-family: 'Times New Roman';">Anaconda</span><span style="font-family: SimSun;">创建虚拟环境，并在虚拟环境中安装依赖库。请在</span><span style="font-family: 'Times New Roman';">conda</span><span style="font-family: SimSun;">环境下进入代码目录<span style="font-family: 'Times New Roman';">./RT_GENE/rt_gene/requirement/</span>，并且输入下面代码即可一键配置好基础环境。</span>
```
conda env create -f environment.yaml
```
</span><span style="font-family: SimSun;"></span>

## <center><span style="font-family: SimHei, 黑体;">3.系统的使用说明</span></center>

### 3.1.系统的硬件要求

#### 3.1.1.双目近景相机
<span style="font-family: SimSun;">双目近景相机采用普通相机与红外相机组合，两者性能需满足：</span>
- <span style="font-family: SimSun;">帧率至少达到</span><span style="font-family: 'Times New Roman';">60FPS</span><span style="font-family: SimSun;">；</span>
- <span style="font-family: SimSun;">普通相机分辨率至少为</span><span style="font-family: 'Times New Roman';">1080p</span><span style="font-family: SimSun;">，红外相机分辨率至少为</span><span style="font-family: 'Times New Roman';">720p</span><span style="font-family: SimSun;">。</span>

<span style="font-family: SimSun;">本系统搭建时使用的双目近景相机为小虎摄像头模组高清</span><span style="font-family: 'Times New Roman';">200</span><span style="font-family: SimSun;">万像素红外摄像头及大疆</span><span style="font-family: 'Times New Roman';">Pocket 3</span><span style="font-family: SimSun;">。</span>

#### 3.1.2.远景相机
<span style="font-family: SimSun;">远景相机通过</span><span style="font-family: 'Times New Roman';">VR</span><span style="font-family: SimSun;">眼镜实现场景显示，</span><span style="font-family: 'Times New Roman';">VR</span><span style="font-family: SimSun;">眼镜性能需满足：</span>
- <span style="font-family: SimSun;">视频流帧率至少达到</span><span style="font-family: 'Times New Roman';">60FPS</span><span style="font-family: SimSun;">；</span>
- <span style="font-family: SimSun;">分辨率至少设置为</span><span style="font-family: 'Times New Roman';">1080p</span><span style="font-family: SimSun;">。</span>

<span style="font-family: SimSun;">本系统搭建时使用的</span><span style="font-family: 'Times New Roman';">VR</span><span style="font-family: SimSun;">眼镜为</span><span style="font-family: 'Times New Roman';">PICO 4</span><span style="font-family: SimSun;">。</span>

### 3.2.系统参数测量及操作流程

#### 3.2.1.第一步：相机标定与环境配置
<span style="font-weight: bold; font-family: SimSun;">设备摆放规范：</span>
- <span style="font-family: SimSun;">摄像头居中固定于显示屏正后方</span>
- <span style="font-family: SimSun;">摄像头光轴保持水平</span>
- <span style="font-family: SimSun;">显示屏平面保持垂直</span>
- <span style="font-family: SimSun;">摄像头光心、显示屏中心、人脸中心三点共线</span>
- <span style="font-family: SimSun;">被试者头部正对摄像头，保持静止姿态（视线可轻微下看）</span>

<span style="font-weight: bold; font-family: SimSun;">必测几何参数（单位：</span><span style="font-weight: bold; font-family: 'Times New Roman';">cm</span><span style="font-weight: bold; font-family: SimSun;">）：</span>
- <span style="font-family: SimSun;">显示屏上沿距桌面高度：</span><span style="font-family: 'Times New Roman';">SCREEN_TOP</span>
- <span style="font-family: SimSun;">显示屏下沿距桌面高度：</span><span style="font-family: 'Times New Roman';">SCREEN_BOTTOM</span>
- <span style="font-family: SimSun;">显示屏物理宽度：</span><span style="font-family: 'Times New Roman';">SCREEN_WIDE</span>
- <span style="font-family: SimSun;">摄像头镜头中心距桌面高度：</span><span style="font-family: 'Times New Roman';">O3_HEIGHT</span>
- <span style="font-family: SimSun;">摄像头到显示屏垂直距离：</span><span style="font-family: 'Times New Roman';">O3_TO_SCREEN</span>
- <span style="font-family: SimSun;">摄像头到人脸中心水平距离：</span><span style="font-family: 'Times New Roman';">O3_TO_FACE</span>

<span style="font-weight: bold; font-family: SimSun;">摄像头配置：</span>
- <span style="font-family: SimSun;">镜像模式：</span><span style="font-family: 'Times New Roman';">MIRROR</span><span style="font-family: SimSun;">（</span><span style="font-family: 'Times New Roman';">1=</span><span style="font-family: SimSun;">启用镜像模式，</span><span style="font-family: 'Times New Roman';">0=</span><span style="font-family: SimSun;">禁用）</span>
- <span style="font-family: SimSun;">人脸取景要求：</span>
  - <span style="font-family: SimSun;">人脸需完整入镜，且尽量充满人脸区域（通过调整焦距实现）</span>
  - <span style="color: red; font-family: SimSun;">焦距固定后禁止变更，否则需重新标定</span><span style="color: red; font-family: 'Times New Roman';">FOCUS</span><span style="color: red; font-family: SimSun;">变量</span>
- <span style="font-family: SimSun;">人像画面边界坐标（像素坐标系）：</span>
  - <span style="font-family: SimSun;">左边界：</span><span style="font-family: 'Times New Roman';">FACE_LEFT</span>
  - <span style="font-family: SimSun;">上边界：</span><span style="font-family: 'Times New Roman';">FACE_TOP</span>
  - <span style="font-family: SimSun;">右边界：</span><span style="font-family: 'Times New Roman';">FACE_RIGHT</span>
  - <span style="font-family: SimSun;">下边界：</span><span style="font-family: 'Times New Roman';">FACE_BOTTOM</span>
  - <span style="font-weight: bold; font-family: SimSun;">示例：</span><span style="font-family: SimSun;">屏幕分辨率</span><span style="font-family: 'Times New Roman';">1920×1080</span><span style="font-family: SimSun;">时，若人脸检测框如图示：</span>
    - <span style="font-family: 'Times New Roman';">FACE_LEFT = 39px</span>
    - <span style="font-family: 'Times New Roman';">FACE_TOP = 39px</span>
    - <span style="font-family: 'Times New Roman';">FACE_RIGHT = 451px</span><span style="font-family: SimSun;">（计算：</span><span style="font-family: 'Times New Roman';">1920 - 1469</span><span style="font-family: SimSun;">）</span>
    - <span style="font-family: 'Times New Roman';">FACE_BOTTOM = 554px</span><span style="font-family: SimSun;">（计算：</span><span style="font-family: 'Times New Roman';">1080 - 526</span><span style="font-family: SimSun;">）</span>

<span style="font-weight: bold; font-family: SimSun;">相机标定流程：</span>
1. <span style="font-weight: bold; font-family: SimSun;">焦距标定与眼部参数计算</span>
   1. <span style="font-family: SimSun;">在后续注视点标定录制过程中，将已知长度</span><span style="font-family: 'Times New Roman';">w</span><span style="font-family: SimSun;">（</span><span style="font-family: 'Times New Roman';">cm</span><span style="font-family: SimSun;">）的标定参照物（建议使用刻度尺）置于前额处</span>
   2. <span style="font-family: SimSun;">录制完成后提取清晰帧，测量参照物对应像素长度</span><span style="font-family: 'Times New Roman';">p</span><span style="font-family: SimSun;">（</span><span style="font-family: 'Times New Roman';">px</span><span style="font-family: SimSun;">）</span>
   3. <span style="font-family: SimSun;">计算焦距参数：</span>
      \[
      \texttt{FOCUS} = \frac{\texttt{O3\_TO\_FACE} \times p}{w}
      \]
   4. <span style="font-family: SimSun;">测量双眼生理参数：</span>
      - <span style="font-family: SimSun;">选取左右眼内外眦（眼角）特征点（即眼白与眼皮交界处）</span>
      - <span style="font-family: SimSun;">测量眼宽（内外眼角）的像素距离</span><span style="font-family: 'Times New Roman';">p_{\text{eye}}</span>
      - <span style="font-family: SimSun;">计算实际眼宽：</span>
        \[
        \texttt{EYE\_WIDTH} = \frac{\texttt{FOCUS} \times w}{p \times p_{\text{eye}}}
        \]
   5. <span style="font-family: SimSun;">设置分辨率压缩系数：</span>
      \[
      \texttt{K\_PX\_COMPRESS} = \frac{\text{视频原始分辨率}}{\text{截图分辨率}}
      \]
      <span style="font-family: SimSun;">（分辨率相同时取</span><span style="font-family: 'Times New Roman';">1.0</span><span style="font-family: SimSun;">）</span>

2. <span style="font-weight: bold; font-family: SimSun;">注视点映射关系标定</span>
   1. <span style="font-family: SimSun;">使用屏幕录制软件同时捕获显示屏内容与摄像头画面</span>
      - <span style="font-family: SimSun;">帧率：≥</span><span style="font-family: 'Times New Roman';">60fps</span>
      - <span style="font-family: SimSun;">输出格式：</span><span style="font-family: 'Times New Roman';">MP4</span>
      - <span style="font-family: SimSun;">人脸区域占比≤</span><span style="font-family: 'Times New Roman';">25\%</span><span style="font-family: SimSun;">屏幕面积</span>
   2. <span style="font-family: SimSun;">运行标定程序</span><span style="font-family: 'Times New Roman';">calibration.html</span><span style="font-family: SimSun;">，按提示完成</span><span style="font-family: 'Times New Roman';">17</span><span style="font-family: SimSun;">点标定流程</span>
   3. <span style="font-family: SimSun;">执行映射关系生成：</span>
      <span style="font-family: SimSun;">注："计算每个标定点的代表坐标"有3种方法：</span><span style="font-family: 'Times New Roman';">{'median', 'kde','weighted'}</span><span style="font-family: SimSun;">；"训练映射模型"有2种模型:</span><span style="font-family: 'Times New Roman';">{'tps' 或 'poly'}</span><span style="font-family: SimSun;">。默认使用</span><span style="font-family: 'Times New Roman';">'kde'+'tps'</span><span style="font-family: SimSun;">，如有需要请在代码中自行更改。</span>
   4. <span style="font-family: SimSun;">输出映射关系文件：</span><span style="font-family: 'Times New Roman';">calibration_result.npy</span>

<span style="font-weight: bold; font-family: SimSun;">示例：</span><span style="font-family: SimSun;">屏幕分辨率</span><span style="font-family: 'Times New Roman';">1920×1080</span><span style="font-family: SimSun;">时，若人脸检测框如图1所示：</span>

<center>
  <img src="./tiaozhanbei/16.png" width="400" alt="人脸检测框">
  <p style="font-family: SimSun;">图1：人脸检测框</p>
</center>

#### 3.2.2.第二步：数据库配置与初始化
<span style="font-family: SimSun;">数据库是系统数据存储与管理的核心组件，在使用系统前需完成数据库的配置与初始化工作。</span>

1. <span style="font-family: SimSun;">数据库连接参数配置</span>

<span style="font-family: SimSun;">需要修改两个文件中的数据库连接参数以匹配本地</span><span style="font-family: 'Times New Roman';">MySQL</span><span style="font-family: SimSun;">环境：</span>

1) <span style="font-family: SimSun;">修改主程序配置文件</span><span style="font-family: 'Times New Roman';">gui_english.py</span><span style="font-family: SimSun;">中的数据库连接参数：</span>

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

2) <span style="font-family: SimSun;">修改数据库设置脚本</span><span style="font-family: 'Times New Roman';">setup_database.py</span><span style="font-family: SimSun;">中的相同连接参数，确保配置一致。</span>

<span style="font-family: SimSun;">参数说明：</span>
- <span style="font-family: 'Times New Roman';">host</span><span style="font-family: SimSun;">: MySQL服务器地址，本地服务器一般为"localhost"</span>
- <span style="font-family: 'Times New Roman';">user</span><span style="font-family: SimSun;">: 数据库登录用户名</span>
- <span style="font-family: 'Times New Roman';">password</span><span style="font-family: SimSun;">: 登录密码（根据实际设置填写）</span>
- <span style="font-family: 'Times New Roman';">database</span><span style="font-family: SimSun;">: 数据库名称，默认为"driver_attention_db"</span>
- <span style="font-family: 'Times New Roman';">port</span><span style="font-family: SimSun;">: 数据库端口，默认3307（根据实际设置修改）</span>

2. <span style="font-family: SimSun;">数据库初始化</span>
<span style="font-family: SimSun;">完成参数配置后，执行以下步骤初始化数据库：</span>

1) <span style="font-family: SimSun;">打开终端，进入程序目录：</span>
```bash
cd /home/public/base_prog
```

2) <span style="font-family: SimSun;">运行数据库设置脚本：</span>
```bash
python setup_database.py
```

3) <span style="font-family: SimSun;">在脚本交互界面中，选择"1"初始化数据库（首次使用），或选择"2"测试数据库连接。</span>

<span style="font-family: SimSun;">数据库初始化将自动创建以下数据表结构：</span>
- <span style="font-family: 'Times New Roman';">drivers</span><span style="font-family: SimSun;">：存储驾驶员基本信息（ID、姓名、驾照类型等）</span>
- <span style="font-family: 'Times New Roman';">experiments</span><span style="font-family: SimSun;">：记录实验概要数据（时间、时长、帧数等）</span>
- <span style="font-family: 'Times New Roman';">attention_data</span><span style="font-family: SimSun;">：保存帧级注意力数据（注视点、瞳孔大小等）</span>

3. <span style="font-family: SimSun;">权限与故障排除</span>
<span style="font-family: SimSun;">确保MySQL用户具有以下权限：</span>
- <span style="font-family: SimSun;">数据库创建权限</span>
- <span style="font-family: SimSun;">表创建与修改权限</span>
- <span style="font-family: SimSun;">数据插入、查询、更新和删除权限</span>

<span style="font-family: SimSun;">常见问题处理：</span>
- <span style="font-family: SimSun;">连接失败：检查MySQL服务是否启动、用户名密码是否正确、防火墙设置</span>
- <span style="font-family: SimSun;">权限错误：确认数据库用户具有足够操作权限</span>
- <span style="font-family: SimSun;">编码问题：确保使用utf8mb4字符集（配置文件已默认设置）</span>

#### 3.2.3.第三步：系统的使用流程说明

<span style="font-family: SimSun;">原项目模型已完成训练，预训练权重无需修改；若重新适配新相机，需根据第一步测量的参数修改对应代码文件（具体修改位置参见代码注释）。</span>

<span style="font-family: SimSun;">修改完成后，运行</span><span style="font-family: 'Times New Roman';">gui.py</span><span style="font-family: SimSun;">文件，打开后界面如图2所示：</span>

<center>
  <img src="./tiaozhanbei/1.png" width="400" alt="系统主界面">
  <p style="font-family: SimSun;">图2：系统主界面</p>
</center>

<span style="font-family: SimSun;">若目录中未找到相机校准文件，系统将弹出提示框，处理过程将使用默认值。此时选择“OK”即可，如图3所示：</span>

<center>
  <img src="./tiaozhanbei/2.png" width="400" alt="相机校准文件缺失提示">
  <p style="font-family: SimSun;">图3：相机校准文件缺失提示</p>
</center>

<span style="font-family: SimSun;">模型加载完成后，将弹出信息框，选择OK确认。确认后可看到完整界面，界面顶部有两个选项卡，分别用于选择视频检测和实时摄像头检测。</span>

<span style="font-weight: bold; font-family: SimSun;">视频处理界面（如图4所示）</span>

<center>
  <img src="./tiaozhanbei/3.png" width="400" alt="视频处理界面">
  <p style="font-family: SimSun;">图4：视频处理界面</p>
</center>

<span style="font-family: SimSun;">视频检测界面由三部分组成：视频处理控制按钮、处理参数显示区域、处理视频显示区域。</span>

1. <span style="font-family: SimSun;">视频控制处理按钮（如图5所示）</span>

<center>
  <img src="./tiaozhanbei/4.png" width="400" alt="视频处理控制按钮">
  <p style="font-family: SimSun;">图5：视频处理控制按钮</p>
</center>

<span style="font-family: SimSun;">上图红框部分为视频处理控制按钮。处理视频前，需上传两个视频：含人脸与场景的视频及对应时间的红外人眼视频。</span>

<span style="font-family: SimSun;">点击“</span><span style="font-family: 'Times New Roman';">Upload Normal Video</span><span style="font-family: SimSun;">”，选择含人脸、场景的视频路径，如图6所示：</span>

<center>
  <img src="./tiaozhanbei/5.png" width="400" alt="选择场景视频路径">
  <p style="font-family: SimSun;">图6：选择场景视频路径</p>
</center>

<span style="font-family: SimSun;">选择完毕后点击OK，所选视频名称将显示在“</span><span style="font-family: 'Times New Roman';">Upload Normal Video</span><span style="font-family: SimSun;">”按钮上。同理，选择并上传红外视频。两个视频均上传完成后，“</span><span style="font-family: 'Times New Roman';">Process Video</span><span style="font-family: SimSun;">”选项将变为绿色，代表可开始处理，如图7所示：</span>

<center>
  <img src="./tiaozhanbei/6.png" width="400" alt="视频上传完成状态">
  <p style="font-family: SimSun;">图7：视频上传完成状态</p>
</center>

<span style="font-family: SimSun;">若需录制处理后的视频片段，可选择界面下半部分的“</span><span style="font-family: 'Times New Roman';">Set Save Path</span><span style="font-family: SimSun;">”以指定录制视频的保存路径，如图8所示：</span>

<center>
  <img src="./tiaozhanbei/7.png" width="400" alt="设置视频保存路径">
  <p style="font-family: SimSun;">图8：设置视频保存路径</p>
</center>

<span style="font-family: SimSun;">选择完毕后将弹出信息框，展示所选路径；同时，“</span><span style="font-family: 'Times New Roman';">Set Save Path</span><span style="font-family: SimSun;">”按钮将显示为保存的目标文件夹。开始处理视频后，点击“</span><span style="font-family: 'Times New Roman';">Start Recording</span><span style="font-family: SimSun;">”开始录制，点击“</span><span style="font-family: 'Times New Roman';">Stop Recording</span><span style="font-family: SimSun;">”结束录制，录制完成后可在对应目录下查看视频文件，如图9所示：</span>

<center>
  <img src="./tiaozhanbei/8.png" width="400" alt="视频录制控制">
  <p style="font-family: SimSun;">图9：视频录制控制</p>
</center>

<span style="font-family: SimSun;">点击“</span><span style="font-family: 'Times New Roman';">Process Video</span><span style="font-family: SimSun;">”按钮开始处理视频，此时需选择相应的驾驶员，如图10所示：</span>

<center>
  <img src="./tiaozhanbei/9.png" width="400" alt="选择驾驶员">
  <p style="font-family: SimSun;">图10：选择驾驶员</p>
</center>

<span style="font-family: SimSun;">若需终止处理，选择“</span><span style="font-family: 'Times New Roman';">Stop Processing</span><span style="font-family: SimSun;">”即可。</span>

2. <span style="font-family: SimSun;">视频显示区域（如图11所示）</span>

<center>
  <img src="./tiaozhanbei/10.png" width="400" alt="视频显示区域">
  <p style="font-family: SimSun;">图11：视频显示区域</p>
</center>

<span style="font-family: SimSun;">检测过程中，视频显示区域将实时展示各物品区域、视线落点及视线向量。</span>

3. <span style="font-family: SimSun;">参数显示区域（如图12所示）</span>

<center>
  <img src="./tiaozhanbei/11.png" width="400" alt="参数显示区域">
  <p style="font-family: SimSun;">图12：参数显示区域</p>
</center>

<span style="font-family: SimSun;">红框部分为参数显示区域，展示的参数包括：</span>
- <span style="font-family: SimSun;">视线向量获取延时（</span><span style="font-family: 'Times New Roman';">Gaze Delay</span><span style="font-family: SimSun;">）</span>
- <span style="font-family: SimSun;">视线落点区域（</span><span style="font-family: 'Times New Roman';">Gaze Region</span><span style="font-family: SimSun;">）</span>
- <span style="font-family: SimSun;">疲劳状态（正常/疲劳）（</span><span style="font-family: 'Times New Roman';">Fatigue Status</span><span style="font-family: SimSun;">）</span>
- <span style="font-family: SimSun;">瞳孔直径（</span><span style="font-family: 'Times New Roman';">Pupil Size</span><span style="font-family: SimSun;">）</span>
- <span style="font-family: SimSun;">视线落点采样率（</span><span style="font-family: 'Times New Roman';">Sampling Rate</span><span style="font-family: SimSun;">）</span>

<span style="font-family: SimSun;">界面最底部有处理进度条，实时展示处理进度。</span>

4. <span style="font-family: SimSun;">数据库界面展示</span>
<span style="font-family: SimSun;">视频处理完成后，实验数据将自动存储至数据库。访问数据库界面时，点击“</span><span style="font-family: 'Times New Roman';">Driver Database</span><span style="font-family: SimSun;">”按钮，系统将弹出数据库窗口，如图13所示：</span>

<center>
  <img src="./tiaozhanbei/12.png" width="400" alt="数据库窗口">
  <p style="font-family: SimSun;">图13：数据库窗口</p>
</center>

<span style="font-family: SimSun;">数据库管理界面功能包括：</span>
- <span style="font-family: SimSun;">驾驶员信息管理：添加、编辑、删除驾驶员信息</span>
- <span style="font-family: SimSun;">搜索功能：支持按ID、姓名、驾照类型搜索</span>
- <span style="font-family: SimSun;">数据导出：将驾驶员列表导出为</span><span style="font-family: 'Times New Roman';">CSV</span><span style="font-family: SimSun;">文件</span>
- <span style="font-family: SimSun;">统计分析：显示驾驶员总数、类型分布等统计信息</span>

<span style="font-family: SimSun;">点击“</span><span style="font-family: 'Times New Roman';">View Experiments</span><span style="font-family: SimSun;">”，可查看选定驾驶员的所有实验记录，如图14所示：</span>

<center>
  <img src="./tiaozhanbei/13.png" width="400" alt="实验数据查看界面">
  <p style="font-family: SimSun;">图14：实验数据查看界面</p>
</center>

<span style="font-family: SimSun;">在实验记录界面中，可查看每次实验的基本信息（日期、时长、总帧数等），点击“</span><span style="font-family: 'Times New Roman';">View Details</span><span style="font-family: SimSun;">”可查看该实验的帧级详细数据，包括注视区域、注视点坐标、瞳孔大小和疲劳状态等信息。</span>

<span style="font-weight: bold; font-family: SimSun;">摄像头实时检测界面（如图15所示）</span>

<center>
  <img src="./tiaozhanbei/14.png" width="400" alt="摄像头实时检测界面">
  <p style="font-family: SimSun;">图15：摄像头实时检测界面</p>
</center>

<span style="font-family: SimSun;">摄像头检测界面与视频录制界面组成相同，差异在于需接入人脸、场景视频流及拍摄人眼的红外视频流。这里推荐使用</span><span style="font-family: 'Times New Roman';">OBS Studio</span><span style="font-family: SimSun;">软件，可以将人脸视频与场景视频结合起来传入系统，录屏期间玩家能看到全部屏幕。</span>

<span style="font-family: SimSun;">点击“</span><span style="font-family: 'Times New Roman';">Scene Video Input</span><span style="font-family: SimSun;">”，输入摄像头编号或视频流</span><span style="font-family: 'Times New Roman';">URL</span><span style="font-family: SimSun;">，如图16所示：</span>

<center>
  <img src="./tiaozhanbei/15.png" width="400" alt="输入摄像头信息">
  <p style="font-family: SimSun;">图16：输入摄像头信息</p>
</center>

<span style="font-family: SimSun;">摄像头选择完毕后，可按照视频处理的功能与流程进行操作。</span>

<span style="font-family: SimSun;">注：若需远程连接</span><span style="font-family: 'Times New Roman';">Linux</span><span style="font-family: SimSun;">系统服务器运行程序，需先安装</span><span style="font-family: 'Times New Roman';">Xming</span><span style="font-family: SimSun;">和</span><span style="font-family: 'Times New Roman';">Putty</span><span style="font-family: SimSun;">以转发图形界面。</span>