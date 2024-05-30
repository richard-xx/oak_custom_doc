# DepthAI 深度查看器

## 介绍
本脚本使用 DepthAI 库与兼容设备进行交互，获取深度信息，并实时显示视差图。它允许用户实时查看深度信息，以及将深度帧保存为 PNG 文件。

## 前提条件和依赖关系
- 兼容 DepthAI 的硬件设备（如 OAK-D 相机）
- Python 3.x 环境
- 库：depthai, opencv-python, numpy

## 安装
安装所需的 Python 库，请执行以下命令：
```bash
pip install depthai opencv-python numpy
```

### 源码
??? note "depth_viewer.py"

    ```python
    --8<-- "src/oak/depth/depth_viewer.py"
    ```

## 用法
在命令行中运行脚本，可以通过参数自定义单目相机的分辨率和帧率：
```bash
python depth_viewer.py --mono_resolution 400p --fps 30
```

## 可用参数
- `-mres`, `--mono_resolution`：单目相机分辨率（高度，单位：像素）
- `-f`, `--fps`：相机帧率
- `-f`, `--filename`：输出深度数据文件名 (不含后缀)
- ... [包括所有其他参数及其解释]

## 保存深度帧
按下 's' 键可以将当前显示的深度帧保存为 PNG 图像，保存位置在脚本文件所在的目录。

## 示例
启动深度查看器并保存深度帧：
```bash
python depth_viewer.py
```
运行后，按 's' 键即可在脚本所在目录下保存名为 "depth.png" 的深度帧。

[//]: # (## 联系方式和贡献)

[//]: # (如有任何疑问或希望贡献代码，请联系项目维护者 [您的姓名]（[您的电子邮件]），或在此项目的 GitHub 页面提交问题或拉取请求。)
