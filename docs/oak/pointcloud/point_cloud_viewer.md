# DepthAI 点云查看器

## 介绍
本脚本允许用户通过兼容 DepthAI 的硬件捕获 3D 点云数据，并使用 Open3D 实现实时可视化。该脚本的一个重要功能是能够将捕获的点云以多种格式保存到磁盘上。

## 前提条件和依赖关系
确保你的系统满足以下要求：

- 兼容 DepthAI 的硬件设备（例如 OAK-D 相机）
- Python 3.x 环境
- 必要的 Python 库：depthai, opencv-python, numpy, open3d

## 安装
在终端或命令提示符中运行以下命令以安装所需库：
```bash
pip install depthai opencv-python numpy open3d
```

### 源码
??? note "point_cloud_viewer.py"

    ```python
    --8<-- "src/oak/pointcloud/point_cloud_viewer.py"
    ```

## 用法
运行脚本并通过命令行参数自定义设置，例如设置相机分辨率和帧率：
```bash
python point_cloud_viewer.py --mono_resolution 400p --color_resolution 1080p --fps 30
```

## 可用参数
- `-mres`, `--mono_resolution`：单目相机分辨率（高度，单位：像素）
- `-cres`, `--color_resolution`：彩色相机分辨率
- `-f`, `--fps`：相机帧率
- `-f`, `--filename`：输出 PLY 文件名 (不含后缀)
- `-suf`， `--suffix`：输出 PLY 后缀名
- ... [包括所有其他参数及其解释]

## 保存点云
脚本提供了保存捕获的点云数据的功能。用户可以通过按 's' 键来触发保存过程，并可以通过 `--output` 参数自定义保存的文件名和 `--pointcloud` 参数来指定文件格式。

支持的点云格式有：

+ PLY（默认）
+ PCD
+ XYZ
+ XYZRGB

## 示例
启动查看器并将点云数据保存为默认的 PLY 格式：
```bash
python point_cloud_viewer.py --output my_point_cloud
```

启动查看器并将点云数据保存为 PCD 格式：
```bash
python point_cloud_viewer.py --output my_point_cloud -p pcd
```
当你按下 's' 键时，点云将保存到脚本文件所在目录下，文件名为 my_point_cloud.pcd。


## 常见问题解答

!!! info "问：如果收到关于缺少 Open3D 的错误怎么办？"

    答：确保您已经使用给出的安装命令安装了所有依赖项。


[//]: # (## 联系方式和贡献)

[//]: # (如有任何问题或希望做出贡献，请联系 [您的姓名]，[您的电邮]。)
