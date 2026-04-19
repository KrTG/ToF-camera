import statistics
import time

import ArducamDepthCamera as ac
import cv2
import numpy as np

from src.tof_camera import TofCamera


camera = TofCamera(frame_timeout=1200)
camera.start()

icpo = cv2.rgbd.RgbdICPOdometry.create(
    cameraMatrix=camera.get_intrinsic_matrix(),
    minDepth=0.2, # meters
    maxDepth=2.4,  # meters
    maxDepthDiff=0.15 # meters
)
print("----ICPO SETTINGS----")
print(f"Camera matrix: {icpo.getCameraMatrix()}")
print(f"Min depth: {icpo.getMinDepth()}")
print(f"Max depth: {icpo.getMaxDepth()}")
print(f"Max depth diff: {icpo.getMaxDepthDiff()}")
print(f"Max points part: {icpo.getMaxPointsPart()}")
print(f"Iter counts: {icpo.getIterationCounts()}")
print(f"Min gradient magnitudes: {icpo.getMinGradientMagnitudes()}")
print(f"Transform type: {icpo.getTransformType()}")
print("--------")

last_frame_time = None
frame_times = []
frame_number = 0
previous_odometry_frame = None
previous_transform = np.eye(4, dtype=np.float64)
global_pose = np.eye(4, dtype=np.float64)
while True:
    amplitude, depth, mask = camera.get_frame_rgbd()

    if amplitude is not None and depth is not None:
        time_now = time.monotonic_ns()

        current_odometry_frame = cv2.rgbd.OdometryFrame.create(
            amplitude, depth, mask, None, frame_number
        )
        icpo.prepareFrameCache(current_odometry_frame, cv2.rgbd.ODOMETRY_FRAME_CACHE_ALL)

        if previous_odometry_frame is not None:
            success, transform = icpo.compute2(
                previous_odometry_frame,
                current_odometry_frame,
                initRt=previous_transform
            )
            if success:
                global_pose = global_pose @ transform
                previous_odometry_frame = current_odometry_frame
                previous_transform = transform
            else:
                print("Tracking lost.")
        else:
            previous_odometry_frame = current_odometry_frame

        if last_frame_time is not None and frame_number > 10:
            frame_time = time_now - last_frame_time
            frame_times.append(frame_time)
        last_frame_time = time_now
        frame_number += 1
    else:
        raise RuntimeError("Frames should not be getting dropped.")

    if (frame_number + 1) % 100 == 0:
        print(f"Frame: {frame_number}")
        print(f"Mean frame time: {statistics.mean(frame_times) / 1000000} ms")
        print(f"Min frame time: {min(frame_times) / 1000000} ms")
        print(f"Max frame time: {max(frame_times) / 1000000} ms")
        print(f"Global pose: {global_pose}")
        frame_times.clear()
