#!/usr/bin/env python3
# coding=utf-8

from pathlib import Path

import cv2
import depthai as dai

root = Path(__file__).parent.resolve()

if compiled := globals().get("__compiled__"):
    root = Path(compiled.containing_dir).resolve()

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and output
cam_rgb = pipeline.create(dai.node.ColorCamera)
video_enc = pipeline.create(dai.node.VideoEncoder)
xout = pipeline.create(dai.node.XLinkOut)

xout.setStreamName("h265")

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.isp.link(xout_rgb.input)

# Properties
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
video_enc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.H265_MAIN)

# Linking
cam_rgb.video.link(video_enc.input)
video_enc.bitstream.link(xout.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the encoded data from the output defined above
    q = device.getOutputQueue(name="h265", maxSize=30, blocking=True)
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=30, blocking=True)

    cv2.namedWindow("rgb", cv2.WINDOW_NORMAL)

    # The .h265 file is a raw stream file (not playable yet)
    with root.joinpath("video.h265").open("wb") as video_file:
        print("Press Ctrl+C to stop encoding...")
        try:
            while True:
                rgb_packet = rgb_queue.tryGet()  # Non-blocking call, will return a new data that has arrived or None otherwise
                if rgb_packet is not None:
                    frame = rgb_packet.getCvFrame()
                    cv2.imshow("rgb", frame)

                key = cv2.waitKey(1) & 0xff
                if key == ord("q"):
                    break

                h265_packet = q.get()  # Blocking call, will wait until a new data has arrived
                h265_packet.getData().tofile(video_file)  # Appends the packet data to the opened file
        except KeyboardInterrupt:
            # Keyboard interrupt (Ctrl + C) detected
            pass

    print("To view the encoded data, convert the stream file (.h265) into a video file (.mp4) using a command below:")
    print("ffmpeg -framerate 30 -i video.h265 -c copy video.mp4")
