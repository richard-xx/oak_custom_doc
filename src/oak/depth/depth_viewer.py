# coding=utf-8
import argparse
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

root = Path(__file__).parent.resolve()

# 解析命令行参数
parser = argparse.ArgumentParser(description="DepthAI Depth Viewer")
parser.add_argument(
    "-mres",
    "--mono_resolution",
    type=str,
    default="400p",
    choices={"480p", "400p", "720p", "800p", "1200p"},
    help="选择单目相机分辨率（高度）。 （默认：%(default)s）",
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
    default="depth",
    help="输出 png 文件名（默认：%(default)s）",
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
MONO_RES = mono_res_opts.get(args.mono_resolution)

FPS = args.fps
FILENAME = args.filename
EXTENDED_DISPARITY = args.extended_disparity
if args.no_extended_disparity:
    EXTENDED_DISPARITY = False
SUBPIXEL = args.subpixel
if args.no_subpixel:
    SUBPIXEL = False
LR_CHECK = args.lr_check


def create_pipeline():
    """创建 DepthAI 流水线"""
    pipeline = dai.Pipeline()

    # 创建相机节点
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    depth = pipeline.create(dai.node.StereoDepth)

    sync = pipeline.create(dai.node.Sync)
    xOut = pipeline.create(dai.node.XLinkOut)
    xOut.input.setBlocking(False)

    # 设置相机属性
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

    # 将节点连接起来
    monoLeft.out.link(depth.left)
    monoRight.out.link(depth.right)
    depth.disparity.link(sync.inputs["disparity"])
    depth.depth.link(sync.inputs["depth"])
    sync.out.link(xOut.input)
    xOut.setStreamName("out")

    return pipeline, depth.initialConfig.getMaxDisparity()


def main():
    """主函数，运行 DepthAI 深度查看器"""
    pipeline, MaxDisparity = create_pipeline()
    # 连接到设备并启动流水线
    with dai.Device(pipeline) as device:
        device.setIrLaserDotProjectorIntensity(0.5)
        device.setIrFloodLightIntensity(0.5)

        # 获取输出队列，用于从上述输出中获取视差帧
        q = device.getOutputQueue(name="out", maxSize=4, blocking=False)

        while True:
            inMessage = q.get()  # 阻塞调用，等待新的数据到达
            inDisparity = inMessage["disparity"]
            frame = inDisparity.getFrame()
            # 为了更好的可视化效果进行归一化
            frame = (frame * (255 / MaxDisparity)).astype(np.uint8)

            cv2.imshow("disparity", frame)

            # 可用的颜色映射：https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
            cv2.imshow("disparity_color", frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

            if key == ord("s"):
                file_path = root.joinpath(f"{FILENAME}.png")
                depth_frame = inMessage["depth"].getFrame()
                cv2.imwrite(file_path.as_posix(), depth_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                print(f"Saved {file_path}")


if __name__ == "__main__":
    main()
