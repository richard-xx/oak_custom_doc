#!/usr/bin/env python3

import argparse
import collections
import contextlib
import math
import time
from datetime import timedelta

import cv2
import depthai as dai

SYNC = 1

parser = argparse.ArgumentParser(epilog="Press C to capture a set of frames.")
parser.add_argument("-f", "--fps", type=float, default=60, help="Camera sensor FPS, applied to all cams")

args = parser.parse_args()

cam_socket_opts = {
    "CAM_A": dai.CameraBoardSocket.CAM_A,
    "CAM_B": dai.CameraBoardSocket.CAM_B,
    "CAM_C": dai.CameraBoardSocket.CAM_C,
    "CAM_D": dai.CameraBoardSocket.CAM_D,
}


def create_pipeline(cam_list):
    # Start defining a pipeline
    pipeline = dai.Pipeline()
    cam = {}
    xout = {}
    for c in cam_list:
        xout[c] = pipeline.create(dai.node.XLinkOut)
        xout[c].setStreamName(c)
        cam[c] = pipeline.create(dai.node.MonoCamera)
        cam[c].setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        # cam[c].setIspScale(1, 3)  # 1920x1080 -> 1280x720
        cam[c].out.link(xout[c].input)
        cam[c].setBoardSocket(cam_socket_opts[c])
        cam[c].setFps(args.fps)
    return pipeline


def draw_text(
    frame,
    text,
    org,
    color=(255, 255, 255),
    bg_color=(128, 128, 128),
    font_scale=0.5,
    thickness=1,
):
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        bg_color,
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


class FPSHandler:
    """
    Class that handles all FPS-related operations.

    Mostly used to calculate different streams FPS, but can also be
    used to feed the video file based on its FPS property, not app performance (this prevents the video from being sent
    to quickly if we finish processing a frame earlier than the next video frame should be consumed)
    """

    _fps_bg_color = (0, 0, 0)
    _fps_color = (255, 255, 255)
    _fps_type = cv2.FONT_HERSHEY_SIMPLEX
    _fps_line_type = cv2.LINE_AA

    def __init__(self, cap=None, max_ticks=100):
        """
        Constructor that initializes the class with a video file object and a maximum ticks amount for FPS calculation

        Args:
            cap (cv2.VideoCapture, Optional): handler to the video file object
            max_ticks (int, Optional): maximum ticks amount for FPS calculation
        """
        self._timestamp = None
        self._start = None
        self._framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None
        self._useCamera = cap is None

        self._iterCnt = 0
        self._ticks = {}

        if max_ticks < 2:  # noqa: PLR2004
            msg = f"Proviced max_ticks value must be 2 or higher (supplied: {max_ticks})"
            raise ValueError(msg)

        self._maxTicks = max_ticks

    def next_iter(self):
        """Marks the next iteration of the processing loop. Will use `time.sleep` method if initialized with video file object"""
        if self._start is None:
            self._start = time.monotonic()

        if not self._useCamera and self._timestamp is not None:
            frame_delay = 1.0 / self._framerate
            delay = (self._timestamp + frame_delay) - time.monotonic()
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
            float: Calculated FPS or `0.0` (default in case of failure)
        """
        if name in self._ticks and len(self._ticks[name]) > 1:
            time_diff = self._ticks[name][-1] - self._ticks[name][0]
            return (len(self._ticks[name]) - 1) / time_diff if time_diff != 0 else 0.0
        return 0.0

    def fps(self):
        """
        Calculates FPS value based on `nextIter` calls, being the FPS of processing loop

        Returns:
            float: Calculated FPS or `0.0` (default in case of failure)
        """
        if self._start is None or self._timestamp is None:
            return 0.0
        time_diff = self._timestamp - self._start
        return self._iterCnt / time_diff if time_diff != 0 else 0.0

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
        frame_fps = f"{name.upper()} FPS: {round(self.tick_fps(name), 1)}"
        # cv2.rectangle(frame, (0, 0), (120, 35), (255, 255, 255), cv2.FILLED)
        draw_text(
            frame,
            frame_fps,
            (5, 15),
            self._fps_color,
            self._fps_bg_color,
        )

        if "nn" in self._ticks:
            draw_text(
                frame,
                frame_fps,
                (5, 30),
                self._fps_color,
                self._fps_bg_color,
            )


# https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack
with contextlib.ExitStack() as stack:
    device_infos = dai.Device.getAllAvailableDevices()

    if len(device_infos) == 0:
        msg = "No devices found!"
        raise RuntimeError(msg)
    else:
        print("Found", len(device_infos), "devices")
    queues = []

    for device_info in device_infos:
        # Note: the pipeline isn't set here, as we don't know yet what device it is.
        # The extra arguments passed are required by the existing overload variants
        openvino_version = dai.OpenVINO.Version.VERSION_2021_4
        usb_speed = dai.UsbSpeed.SUPER_PLUS
        device = stack.enter_context(dai.Device(openvino_version, device_info, usb_speed))
        # Note: currently on POE, DeviceInfo.getMxId() and Device.getMxId() are different!
        print("=== Connected to " + device_info.getMxId())
        mxid = device.getMxId()
        cameras = device.getConnectedCameras()
        usb_speed = device.getUsbSpeed()
        print("   >>> MXID:", mxid)
        print("   >>> Cameras:", *[c.name for c in cameras])
        print("   >>> USB speed:", usb_speed.name)

        cam_list = {c.name for c in cameras}

        # Get a customized pipeline based on identified device type
        device.startPipeline(create_pipeline(cam_list))

        # Output queue will be used to get the rgb frames from the output defined above
        for cam in cam_list:
            queues.append({  # noqa: PERF401
                "queue": device.getOutputQueue(name=cam, maxSize=4, blocking=False),
                "msgs": [],  # Frame msgs
                "mx": device.getMxId(),
                "cam": cam,
            })

    def check_sync(queues, timestamp):
        matching_frames = []
        for q in queues:
            for i, msg in enumerate(q["msgs"]):
                time_diff = abs(msg.getTimestamp() - timestamp)
                # So below 17ms @ 30 FPS => frames are in sync
                if time_diff <= timedelta(milliseconds=math.ceil(500 / args.fps)):
                    matching_frames.append(i)
                    break

        if len(matching_frames) == len(queues):
            # We have all frames synced. Remove the excess ones
            for i, q in enumerate(queues):
                q["msgs"] = q["msgs"][matching_frames[i] :]
            return True
        return False

    fps_handler = FPSHandler()

    while True:
        if SYNC:
            for q in queues:
                new_msg = q["queue"].tryGet()
                if new_msg is not None:
                    q["msgs"].append(new_msg)
                    if check_sync(queues, new_msg.getTimestamp()):
                        print("=" * 50)
                        for q_ in queues:
                            fps_handler.tick(f"{q_['cam']} - {q_['mx']}")
                            pkg = q_["msgs"].pop(0)
                            frame = pkg.getCvFrame()
                            fps_handler.draw_fps(frame, f"{q_['cam']} - {q_['mx']}")
                            draw_text(frame, f"{pkg.getTimestamp()}", (5, 45))
                            cv2.imshow(f"{q_['cam']} - {q_['mx']}", frame)
                            print(f"{q_['cam']} - {q_['mx']}: {pkg.getTimestamp()}")

        else:
            for q in queues:
                new_msg = q["queue"].tryGet()
                if new_msg is not None:
                    fps_handler.tick(f"{q['cam']} - {q['mx']}")
                    frame = new_msg.getCvFrame()
                    fps_handler.draw_fps(frame, f"{q['cam']} - {q['mx']}")
                    draw_text(frame, f"{new_msg.getTimestamp()}", (5, 45))
                    cv2.imshow(f"{q['cam']} - {q['mx']}", frame)

        if cv2.waitKey(1) == ord("q"):
            break
