from threading import Lock
import threading
import time

import cv2
from src.tof_camera import TofCamera

cam = TofCamera()
camera_lock = threading.Lock()
frame_lock = threading.Lock()
frame_amplitude = None
frame_depth = None


def stream_frames(image="amplitude"):
    print("Streaming video")
    global cam
    global amplitude
    global depth

    with camera_lock:
        if not cam.started:
            cam.start()

    while True:
        with frame_lock:
            raw_frame = cam.get_frame_raw()
            if raw_frame is not None:
                amplitude, depth, mask, _time = cam.get_frame_rgbd(raw_frame)
                amplitude = cam.convert_grayscale(amplitude, mask)
                depth = cam.convert_rgb(depth, mask)
            cam.release_frame_raw(raw_frame)

        if image == "amplitude":
            im = amplitude
        elif image == "depth":
            im = depth
        if im is not None:
            imgencode = cv2.imencode(".jpg", im)[1]
            stringData = imgencode.tobytes()
            output = (
                b"--frame\r\n"
                b"Content-Type: text/plain\r\n\r\n" + stringData + b"\r\n"
            )
            yield output
            time.sleep(0.015)
