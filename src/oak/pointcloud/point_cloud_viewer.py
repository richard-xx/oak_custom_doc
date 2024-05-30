# coding=utf-8
import argparse
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
import open3d as o3d

root = Path(__file__).parent.resolve()

# 解析命令行参数
parser = argparse.ArgumentParser(description="DepthAI Point Cloud Viewer")
parser.add_argument(
    "-mres",
    "--mono_resolution",
    type=str,
    default="400p",
    choices={"480p", "400p", "720p", "800p", "1200p"},
    help="选择单目相机分辨率（高度）。 （默认：%(default)s）",
)
parser.add_argument(
    "-cres",
    "--color_resolution",
    default="1080p",
    choices={
        "720p",
        "800p",
        "1080p",
        "1200p",
        "4k",
        "5mp",
        "12mp",
        "13mp",
        "48mp",
        "1352X1012",
        "1440X1080",
        "2024X1520",
        "4000X3000",
        "5312X6000",
    },
    help="选择彩色相机分辨率/高度。 （默认：%(default)s）",
)
parser.add_argument(
    "-ds",
    "--isp_downscale",
    nargs=2,
    default=[1, 3],
    type=int,
    help="将 ISP 输出按此因子降低采样率（默认：%(default)s）",
)

parser.add_argument(
    "-f",
    "--fps",
    type=int,
    default=30,
    help="相机帧率（默认：%(default)s）",
)

parser.add_argument(
    "-e",
    "--extended_disparity",
    default=False,
    action="store_true",
    help="启用扩展视差，最小深度越近，视差范围加倍（从 95 到 190）（默认：%(default)s）",
)
parser.add_argument(
    "-ne",
    "--no_extended_disparity",
    default=False,
    action="store_true",
    help="禁用扩展视差，最小深度越近，视差范围加倍（从 95 到 190）（默认：%(default)s）",
)
parser.add_argument(
    "-sub",
    "--subpixel",
    default=True,
    action="store_true",
    help="使用亚像素插值（默认：%(default)s）",
)
parser.add_argument(
    "-nsub",
    "--no_subpixel",
    action="store_true",
    help="禁用亚像素插值（默认：%(default)s）",
)
parser.add_argument(
    "-l",
    "--lr_check",
    # default=True,
    action="store_false",
    help="左/右视差检查（默认：%(default)s）",
)
parser.add_argument(
    "-file",
    "--filename",
    type=str,
    default="point_cloud",
    help="输出 PLY 文件名（默认：%(default)s）",
)
parser.add_argument(
    "-s",
    "--sparse",
    action="store_true",
    help="启用或禁用稀疏点云计算（默认：%(default)s）",
)
parser.add_argument(
    "-suf",
    "--suffix",
    default="ply",
    choices={"ply", "pcd", "xyz", "xyzrgb"},
    help="保存点云文件后缀（默认：%(default)s）",
)
args = parser.parse_args()

# 定义单目和彩色相机的分辨率
mono_res_opts = {
    "400p": dai.MonoCameraProperties.SensorResolution.THE_400_P,
    "480p": dai.MonoCameraProperties.SensorResolution.THE_480_P,
    "720p": dai.MonoCameraProperties.SensorResolution.THE_720_P,
    "800p": dai.MonoCameraProperties.SensorResolution.THE_800_P,
    "1200p": dai.MonoCameraProperties.SensorResolution.THE_1200_P,
}

color_res_opts = {
    "720p": dai.ColorCameraProperties.SensorResolution.THE_720_P,
    "800p": dai.ColorCameraProperties.SensorResolution.THE_800_P,
    "1080p": dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    "1200p": dai.ColorCameraProperties.SensorResolution.THE_1200_P,
    "4k": dai.ColorCameraProperties.SensorResolution.THE_4_K,
    "5mp": dai.ColorCameraProperties.SensorResolution.THE_5_MP,
    "12mp": dai.ColorCameraProperties.SensorResolution.THE_12_MP,
    "13mp": dai.ColorCameraProperties.SensorResolution.THE_13_MP,
    "48mp": dai.ColorCameraProperties.SensorResolution.THE_48_MP,
    "1352X1012": dai.ColorCameraProperties.SensorResolution.THE_1352X1012,
    "1440X1080": dai.ColorCameraProperties.SensorResolution.THE_1440X1080,
    "2024X1520": dai.ColorCameraProperties.SensorResolution.THE_2024X1520,
    "4000X3000": dai.ColorCameraProperties.SensorResolution.THE_4000X3000,
    "5312X6000": dai.ColorCameraProperties.SensorResolution.THE_5312X6000,
}

MONO_RES = mono_res_opts.get(args.mono_resolution)
COLOR_RES = color_res_opts.get(args.color_resolution)

FPS = args.fps
SPARSE = args.sparse
SUFFIX = args.suffix
FILENAME = args.filename
NUMERATOR, DENOMINATOR = args.isp_downscale
EXTENDED_DISPARITY = args.extended_disparity
if args.no_extended_disparity:
    EXTENDED_DISPARITY = False
SUBPIXEL = args.subpixel
if args.no_subpixel:
    SUBPIXEL = False

LR_CHECK = args.lr_check


class FPSCounter:
    def __init__(self):
        self.frameCount = 0
        self.fps = 0
        self.startTime = time.time()

    def tick(self):
        self.frameCount += 1
        if self.frameCount % 10 == 0:
            elapsedTime = time.time() - self.startTime
            self.fps = self.frameCount / elapsedTime
            self.frameCount = 0
            self.startTime = time.time()
        return self.fps


def create_pipeline():
    """创建 DepthAI 流水线"""
    pipeline = dai.Pipeline()
    # 创建相机节点
    camRgb = pipeline.create(dai.node.ColorCamera)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    # 创建深度估计和点云节点
    depth = pipeline.create(dai.node.StereoDepth)
    pointcloud = pipeline.create(dai.node.PointCloud)
    # 创建同步节点和输出节点
    sync = pipeline.create(dai.node.Sync)
    xOut = pipeline.create(dai.node.XLinkOut)
    xOut.input.setBlocking(False)

    # 设置相机属性
    camRgb.setResolution(COLOR_RES)
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    camRgb.setIspScale(NUMERATOR, DENOMINATOR)
    camRgb.setFps(FPS)

    monoLeft.setResolution(MONO_RES)
    monoLeft.setCamera("left")
    monoLeft.setFps(FPS)
    monoRight.setResolution(MONO_RES)
    monoRight.setCamera("right")
    monoRight.setFps(FPS)

    # 设置深度估计属性
    depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    depth.setLeftRightCheck(LR_CHECK)
    depth.setExtendedDisparity(EXTENDED_DISPARITY)
    depth.setSubpixel(SUBPIXEL)
    depth.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    depth.setOutputSize(*camRgb.getIspSize())

    # 设置点云属性
    pointcloud.initialConfig.setSparse(SPARSE)

    # 将节点连接起来
    monoLeft.out.link(depth.left)
    monoRight.out.link(depth.right)
    depth.depth.link(pointcloud.inputDepth)
    camRgb.isp.link(sync.inputs["rgb"])
    pointcloud.outputPointCloud.link(sync.inputs["pcl"])
    sync.out.link(xOut.input)
    xOut.setStreamName("out")

    return pipeline


def main():  # noqa: PLR0914, PLR0915
    """主函数，运行 DepthAI 点云查看器"""
    pipeline = create_pipeline()
    with dai.Device(pipeline) as device:
        print("DepthAI 点云查看器已启动！")
        print("按 `q`键退出，按`s` 键保存点云数据。")

        isRunning = True

        def key_callback(vis, action, mods):
            global isRunning  # noqa: PLW0603
            if action == 0:
                isRunning = False

        q = device.getOutputQueue(name="out", maxSize=4, blocking=False)
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        vis.register_key_action_callback(81, key_callback)
        pcd = o3d.geometry.PointCloud()
        coordinateFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000, origin=[0, 0, 0])
        vis.add_geometry(coordinateFrame)

        first = True
        fpsCounter = FPSCounter()
        while isRunning:
            inMessage = q.get()
            inColor = inMessage["rgb"]
            inPointCloud = inMessage["pcl"]
            if inColor is not None:
                cvColorFrame = inColor.getCvFrame()
                # 将帧转换为 RGB 格式
                cvRGBFrame = cv2.cvtColor(cvColorFrame, cv2.COLOR_BGR2RGB)
                fps = fpsCounter.tick()
                # 在帧上显示 FPS
                cv2.putText(
                    cvColorFrame,
                    f"FPS: {fps:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("color", cvColorFrame)

            if inPointCloud:
                time.time()
                points = inPointCloud.getPoints().astype(np.float64)
                pcd.points = o3d.utility.Vector3dVector(points)
                colors = (cvRGBFrame.reshape(-1, 3) / 255.0).astype(np.float64)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                if first:
                    vis.add_geometry(pcd)
                    first = False
                else:
                    vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                file_path = root.joinpath(f"{FILENAME}.{SUFFIX}")
                o3d.io.write_point_cloud(file_path.as_posix(), pcd)
                print(f"将点云保存为 {file_path}")

        cv2.destroyAllWindows()
        vis.destroy_window()


if __name__ == "__main__":
    main()
