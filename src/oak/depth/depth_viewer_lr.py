#!/usr/bin/env python3
# coding=utf-8
import argparse
import collections
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np

CAM_GROUPS = ["LC", "LR", "CR"]

root = Path(__file__).parent.resolve()

# 解析命令行参数
parser = argparse.ArgumentParser(description="DepthAI Depth Viewer For LR")
parser.add_argument(
    "-sp",
    "--stereo_pair",
    default="LR",
    choices=CAM_GROUPS,
    help="相机对（LR、LC、CR），`L` 表示 左相机，`R` 表示 右相机，`C` 表示 中间相机（默认：%(default)s）",
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
    default="depth_lr",
    help="输出 png 文件名（默认：%(default)s）",
)

args = parser.parse_args()

FPS = args.fps
FILENAME = args.filename
EXTENDED_DISPARITY = args.extended_disparity
if args.no_extended_disparity:
    EXTENDED_DISPARITY = False
SUBPIXEL = args.subpixel
if args.no_subpixel:
    SUBPIXEL = False
LR_CHECK = args.lr_check

CAM_GROUP = args.stereo_pair
calculation_algorithm = dai.SpatialLocationCalculatorAlgorithm.AVERAGE
top_left = dai.Point2f(0.4, 0.4)
bottom_right = dai.Point2f(0.6, 0.6)
config = dai.SpatialLocationCalculatorConfigData()


class FPSHandler:
    """
    处理所有 FPS 相关操作的类。

    主要用于计算不同流的 FPS，但也可用于根据视频文件的 FPS 属性而不是应用程序性能来提供视频文件（如果我们比下一个视频帧早完成处理一帧，这会阻止视频快速发送）被消耗）
    """

    _fps_bg_color = (0, 0, 0)
    _fps_color = (255, 255, 255)
    _fps_type = cv2.FONT_HERSHEY_SIMPLEX
    _fps_line_type = cv2.LINE_AA

    def __init__(self, cap=None, maxTicks=100):
        self._timestamp = None
        self._start = None
        self._framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None
        self._useCamera = cap is None

        self._iterCnt = 0
        self._ticks = {}

        if maxTicks < 2:  # noqa: PLR2004
            msg = f"Provided maxTicks value must be 2 or higher (supplied: {maxTicks})"
            raise ValueError(msg)

        self._maxTicks = maxTicks

    def next_iter(self):
        """Marks the next iteration of the processing loop. Will use :obj:`time.sleep` method if initialized with video file object"""
        if self._start is None:
            self._start = time.monotonic()

        if not self._useCamera and self._timestamp is not None:
            frameDelay = 1.0 / self._framerate
            delay = (self._timestamp + frameDelay) - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        self._timestamp = time.monotonic()
        self._iterCnt += 1

    def tick(self, name):
        """
        Marks a point in time for specified name

        Args:
            name (str): Specifies timestamp name
        """
        if name not in self._ticks:
            self._ticks[name] = collections.deque(maxlen=self._maxTicks)
        self._ticks[name].append(time.monotonic())

    def tick_fps(self, name):
        """
        Calculates the FPS based on specified name

        Args:
            name (str): Specifies timestamps' name

        Returns:
            float: Calculated FPS or :code:`0.0` (default in case of failure)
        """
        if name in self._ticks and len(self._ticks[name]) > 1:
            timeDiff = self._ticks[name][-1] - self._ticks[name][0]
            return (len(self._ticks[name]) - 1) / timeDiff if timeDiff != 0 else 0.0
        return 0.0

    def fps(self):
        """
        Calculates FPS value based on :func:`nextIter` calls, being the FPS of processing loop

        Returns:
            float: Calculated FPS or :code:`0.0` (default in case of failure)
        """
        if self._start is None or self._timestamp is None:
            return 0.0
        timeDiff = self._timestamp - self._start
        return self._iterCnt / timeDiff if timeDiff != 0 else 0.0

    def print_status(self):
        """Prints total FPS for all names stored in :func:`tick` calls"""
        print("=== TOTAL FPS ===")
        for name in self._ticks:
            print(f"[{name}]: {self.tick_fps(name):.1f}")

    def draw_fps(self, frame, name):
        """
        Draws FPS values on requested frame, calculated based on specified name

        Args:
            frame (numpy.ndarray): Frame object to draw values on
            name (str): Specifies timestamps' name
        """
        frameFps = f"{name.upper()} FPS: {round(self.tick_fps(name), 1)}"
        # cv2.rectangle(frame, (0, 0), (120, 35), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, frameFps, (5, 15), self._fps_type, 0.5, self._fps_bg_color, 4, self._fps_line_type)
        cv2.putText(frame, frameFps, (5, 15), self._fps_type, 0.5, self._fps_color, 1, self._fps_line_type)

        if "nn" in self._ticks:
            cv2.putText(
                frame,
                f"NN FPS:  " f"{round(self.tick_fps('nn'), 1)}",
                (5, 30),
                self._fps_type,
                0.5,
                self._fps_bg_color,
                4,
                self._fps_line_type,
            )
            cv2.putText(
                frame,
                f"NN FPS:  " f"{round(self.tick_fps('nn'), 1)}",
                (5, 30),
                self._fps_type,
                0.5,
                self._fps_color,
                1,
                self._fps_line_type,
            )


def create_pipeline():
    global calculation_algorithm, config  # noqa: PLW0602

    # Create pipeline
    pipeline = dai.Pipeline()

    # Define sources and outputs
    left = pipeline.create(dai.node.ColorCamera)
    right = pipeline.create(dai.node.ColorCamera)

    stereo = pipeline.create(dai.node.StereoDepth)
    spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

    xoutSpatialData = pipeline.create(dai.node.XLinkOut)
    xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

    sync = pipeline.create(dai.node.Sync)
    xOut = pipeline.create(dai.node.XLinkOut)
    xOut.setStreamName("out")
    xOut.input.setBlocking(False)

    xoutSpatialData.setStreamName("spatialData")
    xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

    # Properties
    left.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
    left.setIspScale(1, 3)
    left.setFps(30)
    right.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
    right.setIspScale(1, 3)
    right.setFps(30)

    if CAM_GROUP == "LC":
        left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        right.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    elif CAM_GROUP == "LR":
        left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    elif CAM_GROUP == "RC":
        left.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
    # LR-check is required for depthQueueData alignment
    stereo.setLeftRightCheck(LR_CHECK)
    stereo.setExtendedDisparity(EXTENDED_DISPARITY)
    stereo.setSubpixel(SUBPIXEL)

    # stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    # Config
    config.depthThresholds.lowerThreshold = 0
    config.depthThresholds.upperThreshold = 100_000  # mm
    config.calculationAlgorithm = calculation_algorithm
    config.roi = dai.Rect(top_left, bottom_right)

    spatialLocationCalculator.inputConfig.setWaitForMessage(False)
    spatialLocationCalculator.initialConfig.addROI(config)

    # Linking
    stereo.syncedRight.link(sync.inputs["image"])

    left.isp.link(stereo.left)
    right.isp.link(stereo.right)

    stereo.disparity.link(sync.inputs["disparity"])
    stereo.depth.link(spatialLocationCalculator.inputDepth)
    spatialLocationCalculator.passthroughDepth.link(sync.inputs["depth"])

    spatialLocationCalculator.out.link(xoutSpatialData.input)

    xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

    return pipeline, stereo.initialConfig.getMaxDisparity()


def check_input(roi, frame, DELTA=5):
    """Check if input is ROI or point. If point, convert to ROI"""
    # Convert to a numpy array if input is a list
    if isinstance(roi, list):
        roi = np.array(roi)

    # Limit the point so ROI won't be outside the frame
    if roi.shape in {(2,), (2, 1)}:
        roi = np.hstack([roi, np.array([[-DELTA, -DELTA], [DELTA, DELTA]])])
    elif roi.shape in {(4,), (4, 1)}:
        roi = np.array(roi)

    roi.clip([DELTA, DELTA], [frame.shape[1] - DELTA, frame.shape[0] - DELTA])

    return roi / frame.shape[1::-1]


def click_and_crop(event, x, y, flags, param):
    """
    单击鼠标左键时记录起始 (x, y) 坐标，释放鼠标左键时记录结束 (x, y) 坐标。以 numpy 数组形式返回坐标。

    Args:
        event (int): 鼠标事件类型。
        x (int): 鼠标事件的 x 坐标。
        y (int): 鼠标事件的 y 坐标。
        flags (int): 鼠标事件的任何相关标志。
        param (object): 鼠标事件的任何相关参数。

    Returns:
        numpy.ndarray: 鼠标事件的坐标作为 numpy 数组。
    """
    # grab references to the global variables
    global ref_pt, click_roi  # noqa: PLW0603
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_pt = [(x, y)]
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        ref_pt.append((x, y))
        ref_pt = np.array(ref_pt)
        click_roi = np.array([np.min(ref_pt, axis=0), np.max(ref_pt, axis=0)])


def run():  # noqa: PLR0912, PLR0914, PLR0915, C901
    global ref_pt, click_roi, calculation_algorithm, config  # noqa: PLW0603, PLW0602
    CALCULATION_ALGORITHMS = {
        ord("1"): dai.SpatialLocationCalculatorAlgorithm.MEAN,
        ord("2"): dai.SpatialLocationCalculatorAlgorithm.MIN,
        ord("3"): dai.SpatialLocationCalculatorAlgorithm.MAX,
        ord("4"): dai.SpatialLocationCalculatorAlgorithm.MODE,
        ord("5"): dai.SpatialLocationCalculatorAlgorithm.MEDIAN,
    }

    # Connect to device and start pipeline
    with dai.Device() as device:
        pipeline, maxDisparity = create_pipeline()
        device.startPipeline(pipeline)

        frameRgb = None
        frameDisp = None
        depthDatas = []
        stepSize = 0.01
        newConfig = False

        # Configure windows; trackbar adjusts blending ratio of rgb/depthQueueData
        rgbWindowName = "image"
        depthWindowName = "depthQueueData"
        cv2.namedWindow(rgbWindowName)
        cv2.namedWindow(depthWindowName)

        cv2.setMouseCallback(rgbWindowName, click_and_crop)
        cv2.setMouseCallback(depthWindowName, click_and_crop)

        print("Use WASD keys to move ROI!")

        spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
        spatialDataQueue = device.getOutputQueue("spatialData")
        q = device.getOutputQueue(name="out", maxSize=4, blocking=False)

        def draw_text(frame, text, org, color=(255, 255, 255), thickness=1):
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness + 3, cv2.LINE_AA)
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness, cv2.LINE_AA)

        def draw_rect(frame, topLeft, bottomRight, color=(255, 255, 255), thickness=1):
            cv2.rectangle(frame, topLeft, bottomRight, (0, 0, 0), thickness + 3)
            cv2.rectangle(frame, topLeft, bottomRight, color, thickness)

        def draw_spatial_locations(frame, spatialLocations):
            for depthData in spatialLocations:
                roi = depthData.config.roi
                roi = roi.denormalize(width=frame.shape[1], height=frame.shape[0])
                xmin = int(roi.topLeft().x)
                ymin = int(roi.topLeft().y)
                xmax = int(roi.bottomRight().x)
                ymax = int(roi.bottomRight().y)

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 0), 4)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
                draw_rect(
                    frame,
                    (xmin, ymin),
                    (xmax, ymax),
                )
                draw_text(
                    frame,
                    f"X: {int(depthData.spatialCoordinates.x)} mm",
                    (xmin + 10, ymin + 20),
                )
                draw_text(
                    frame,
                    f"Y: {int(depthData.spatialCoordinates.y)} mm",
                    (xmin + 10, ymin + 35),
                )
                draw_text(
                    frame,
                    f"Z: {int(depthData.spatialCoordinates.z)} mm",
                    (xmin + 10, ymin + 50),
                )

        fps = FPSHandler()
        while not device.isClosed():
            inMessage = q.tryGet()
            spatialData = spatialDataQueue.tryGet()

            if spatialData is not None:
                depthDatas = spatialData.getSpatialLocations()

            if inMessage is not None:
                imageData = inMessage["image"]
                fps.tick("image")

                frameRgb = imageData.getCvFrame()
                fps.draw_fps(frameRgb, "image")
                draw_spatial_locations(frameRgb, depthDatas)

                dispData = inMessage["disparity"]
                fps.tick("dispData")
                frameDisp = dispData.getFrame()
                frameDisp = (frameDisp * (255 / maxDisparity)).astype(np.uint8)
                frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_JET)
                frameDisp = np.ascontiguousarray(frameDisp)
                draw_spatial_locations(frameDisp, depthDatas)
                fps.draw_fps(frameDisp, "dispData")

                # Blend when both received
            if frameRgb is not None and frameDisp is not None:
                if click_roi is not None:
                    (
                        [top_left.x, top_left.y],
                        [
                            bottom_right.x,
                            bottom_right.y,
                        ],
                    ) = check_input(click_roi, frameRgb)
                    click_roi = None
                    newConfig = True

                cv2.imshow(depthWindowName, frameDisp)
                cv2.imshow(rgbWindowName, frameRgb)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            if key == ord("w"):
                if top_left.y - stepSize >= 0:
                    top_left.y -= stepSize
                    bottom_right.y -= stepSize
                    newConfig = True
            elif key == ord("a"):
                if top_left.x - stepSize >= 0:
                    top_left.x -= stepSize
                    bottom_right.x -= stepSize
                    newConfig = True
            elif key == ord("s"):
                if bottom_right.y + stepSize <= 1:
                    top_left.y += stepSize
                    bottom_right.y += stepSize
                    newConfig = True
            elif key == ord("d"):
                if bottom_right.x + stepSize <= 1:
                    top_left.x += stepSize
                    bottom_right.x += stepSize
                    newConfig = True

            elif key in CALCULATION_ALGORITHMS:
                calculation_algorithm = CALCULATION_ALGORITHMS[key]
                print(f"Switching calculation algorithm to {calculation_algorithm.name}!")
                newConfig = True

            elif key == ord("s"):
                file_path = root.joinpath(f"{FILENAME}.png")
                depth_frame = inMessage["depth"].getFrame()
                cv2.imwrite(file_path.as_posix(), depth_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                print(f"Saved {file_path}")

            if newConfig:
                # config.depthThresholds.lowerThreshold = 0
                # config.depthThresholds.upperThreshold = 10000
                config.roi = dai.Rect(top_left, bottom_right)
                config.calculationAlgorithm = calculation_algorithm

                cfg = dai.SpatialLocationCalculatorConfig()
                cfg.addROI(config)
                spatialCalcConfigInQueue.send(cfg)
                newConfig = False


if __name__ == "__main__":
    ref_pt = None
    click_roi = None
    run()
