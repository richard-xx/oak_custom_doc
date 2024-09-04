#!/usr/bin/env python3

import time

import cv2
import depthai as dai
import open3d as o3d

COLOR = True

lrcheck = True  # Better handling for occlusions
extended = False  # Closer-in minimum depth, disparity range is doubled
subpixel = True  # Better accuracy for longer distance, fractional disparity 32-levels
# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

print("StereoDepth config options:")
print("    Left-Right check:  ", lrcheck)
print("    Extended disparity:", extended)
print("    Subpixel:          ", subpixel)
print("    Median filtering:  ", median)

pipeline = dai.Pipeline()

mono_left = pipeline.create(dai.node.MonoCamera)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)

mono_right = pipeline.create(dai.node.MonoCamera)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo = pipeline.createStereoDepth()
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(median)
# stereo.initialConfig.setConfidenceThreshold(255)

stereo.setLeftRightCheck(lrcheck)
stereo.setExtendedDisparity(extended)
stereo.setSubpixel(subpixel)
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

config = stereo.initialConfig.get()
config.postProcessing.speckleFilter.enable = False
config.postProcessing.speckleFilter.speckleRange = 50
config.postProcessing.temporalFilter.enable = True
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.thresholdFilter.minRange = 400
config.postProcessing.thresholdFilter.maxRange = 200000
config.postProcessing.decimationFilter.decimationFactor = 1
stereo.initialConfig.set(config)

xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# xout_disparity = pipeline.createXLinkOut()
# xout_disparity.setStreamName('disparity')
# stereo.disparity.link(xout_disparity.input)

xout_colorize = pipeline.createXLinkOut()
xout_colorize.setStreamName("colorize")
xout_rect_left = pipeline.createXLinkOut()
xout_rect_left.setStreamName("rectified_left")
xout_rect_right = pipeline.createXLinkOut()
xout_rect_right.setStreamName("rectified_right")

if COLOR:
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setIspScale(1, 3)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam_rgb.initialControl.setManualFocus(130)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    cam_rgb.isp.link(xout_colorize.input)
else:
    stereo.rectifiedRight.link(xout_colorize.input)

stereo.rectifiedLeft.link(xout_rect_left.input)
stereo.rectifiedRight.link(xout_rect_right.input)


class HostSync:
    def __init__(self):
        self.arrays = {}

    def add_msg(self, name, msg):
        if name not in self.arrays:
            self.arrays[name] = []
        # Add msg to array
        self.arrays[name].append({"msg": msg, "seq": msg.getSequenceNum()})

        synced = {}
        for arr in self.arrays.values():
            for obj in arr:
                if msg.getSequenceNum() == obj["seq"]:
                    synced[name] = obj["msg"]
                    break
        # If there are 5 (all) synced msgs, remove all old msgs
        # and return synced msgs
        if len(synced) == 4:  # color, left, right, depth, nn  # noqa: PLR2004
            # Remove old msgs
            for arr in self.arrays.values():
                for obj in arr:
                    if obj["seq"] < msg.getSequenceNum():
                        arr.remove(obj)  # noqa: B909
                    else:
                        break
            return synced
        return False


with dai.Device(pipeline) as device:
    device.setIrLaserDotProjectorBrightness(1200)
    qs = []
    qs.extend((
        device.getOutputQueue("depth", maxSize=1, blocking=False),
        device.getOutputQueue("colorize", maxSize=1, blocking=False),
        device.getOutputQueue("rectified_left", maxSize=1, blocking=False),
        device.getOutputQueue("rectified_right", maxSize=1, blocking=False),
    ))

    try:
        from projector_3d import PointCloudVisualizer
    except ImportError as e:
        msg = f"\033[1;5;31mError occured when importing PCL projector: {e}. Try disabling the point cloud \033[0m "
        raise ImportError(msg) from e

    calib_data = device.readCalibration()
    if COLOR:
        w, h = cam_rgb.getIspSize()
        intrinsics = calib_data.getCameraIntrinsics(dai.CameraBoardSocket.RGB, dai.Size2f(w, h))
    else:
        w, h = mono_right.getResolutionSize()
        intrinsics = calib_data.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, dai.Size2f(w, h))
    pcl_converter = PointCloudVisualizer(intrinsics, w, h)

    serial_no = device.getMxId()
    sync = HostSync()
    depth_vis, color, rect_left, rect_right = None, None, None, None

    while True:
        for q in qs:
            new_msg = q.tryGet()
            if new_msg is not None:
                msgs = sync.add_msg(q.getName(), new_msg)
                if msgs:
                    depth = msgs["depth"].getFrame()
                    color = msgs["colorize"].getCvFrame()
                    rectified_left = msgs["rectified_left"].getCvFrame()
                    rectified_right = msgs["rectified_right"].getCvFrame()
                    depth_vis = cv2.normalize(depth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                    depth_vis = cv2.equalizeHist(depth_vis)
                    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_HOT)
                    cv2.imshow("depth", depth_vis)
                    cv2.imshow("color", color)
                    cv2.imshow("rectified_left", rectified_left)
                    cv2.imshow("rectified_right", rectified_right)
                    rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                    pcl_converter.rgbd_to_projection(depth, rgb)
                    pcl_converter.visualize_pcd()

        key = cv2.waitKey(1)
        if key == ord("s"):
            timestamp = str(int(time.time()))
            cv2.imwrite(f"{serial_no}_{timestamp}_depth.png", depth_vis)
            cv2.imwrite(f"{serial_no}_{timestamp}_color.png", color)
            cv2.imwrite(f"{serial_no}_{timestamp}_rectified_left.png", rectified_left)
            cv2.imwrite(f"{serial_no}_{timestamp}_rectified_right.png", rectified_right)
            o3d.io.write_point_cloud(f"{serial_no}_{timestamp}.pcd", pcl_converter.pcl, compressed=True)
        elif key == ord("q"):
            break
