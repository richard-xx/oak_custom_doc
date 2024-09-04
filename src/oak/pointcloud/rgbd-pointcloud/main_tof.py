#!/usr/bin/env python3

import time

import cv2
import depthai as dai
import open3d as o3d

pipeline = dai.Pipeline()

mono_left = pipeline.create(dai.node.ColorCamera)
mono_left.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_left.setIspScale(1, 2)

tof = pipeline.create(dai.node.ToF)
cam_tof = pipeline.create(dai.node.Camera)
image_align = pipeline.create(dai.node.ImageAlign)

xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")

# ToF settings
cam_tof.setFps(15)
cam_tof.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
cam_tof.setBoardSocket(dai.CameraBoardSocket.CAM_A)

sync = pipeline.create(dai.node.Sync)


# Image align
image_align.setOutputSize(640, 400)

# Linking
cam_tof.raw.link(tof.input)
tof.depth.link(image_align.input)

mono_left.isp.link(image_align.inputAlignTo)
image_align.outputAligned.link(sync.inputs["depth"])

mono_left.video.link(sync.inputs["colorize"])

out = pipeline.create(dai.node.XLinkOut)

out.setStreamName("out")
sync.out.link(out.input)


with dai.Device(pipeline) as device:
    # device.setIrLaserDotProjectorBrightness(1200)
    q = device.getOutputQueue("out", maxSize=1, blocking=False)

    try:
        from projector_3d import PointCloudVisualizer
    except ImportError as e:
        msg = f"\033[1;5;31mError occured when importing PCL projector: {e}. Try disabling the point cloud \033[0m "
        raise ImportError(msg) from e

    calib_data = device.readCalibration()
    w, h = mono_left.getIspSize()
    intrinsics = calib_data.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, dai.Size2f(w, h))
    pcl_converter = PointCloudVisualizer(intrinsics, w, h)

    serial_no = device.getMxId()
    depth_vis, color, rect_left, rect_right = None, None, None, None

    while True:
        new_msg = q.tryGet()
        if new_msg is not None:
            depth = new_msg["depth"].getFrame()
            color = new_msg["colorize"].getCvFrame()
            depth_vis = cv2.normalize(depth, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depth_vis = cv2.equalizeHist(depth_vis)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_HOT)
            cv2.imshow("depth", depth_vis)
            cv2.imshow("color", color)
            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            pcl_converter.rgbd_to_projection(depth, rgb)
            pcl_converter.visualize_pcd()

        key = cv2.waitKey(1)
        if key == ord("s"):
            timestamp = str(int(time.time()))
            cv2.imwrite(f"{serial_no}_{timestamp}_depth.png", depth_vis)
            cv2.imwrite(f"{serial_no}_{timestamp}_color.png", color)
            o3d.io.write_point_cloud(f"{serial_no}_{timestamp}.pcd", pcl_converter.pcl, compressed=True)
        elif key == ord("q"):
            break
