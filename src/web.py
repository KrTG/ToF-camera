from collections import deque
import json
from statistics import mean
import threading
import time
from multiprocessing.spawn import prepare
from typing import Optional, Tuple

import cv2

from estimator import (CameraThread, ComputeThread, PipelineThread,
                       PrepareFrameThread, get_rotation_degrees,
                       get_translation)
from src import conf
from src.icpo import IcpOdometry
from src.tof_camera import TofCamera


class FrameSaverThread(PipelineThread):
    def __init__(self, camera_thread: CameraThread, camera: TofCamera):
        super().__init__()
        self.camera_thread = camera_thread
        self.camera = camera
        self.cached_frame = None

    def run(self):
        self.running = True
        try:
            while self.running:
                frame = self.camera_thread.get_frame()
                if frame is None:
                    continue

                # Save the frame globally
                amplitude, depth, mask, _ = frame
                amplitude = self.camera.convert_grayscale(amplitude, mask)
                depth = self.camera.convert_rgb(depth, mask)

                with self.condition:
                    self.cached_frame = amplitude, depth
        finally:
            pass

    def get_frame(self) -> Optional[Tuple]:
        with self.condition:
            return self.cached_frame


class OdometrySaverThread(PipelineThread):
    def __init__(self, compute_thread: PipelineThread):
        super().__init__()
        self.compute_thread = compute_thread

        self.cached_frame = None
        self.prep_times = deque(maxlen=100)
        self.cache_times = deque(maxlen=100)
        self.compute_times = deque(maxlen=100)

    def run(self):
        self.running = True
        try:
            while self.running:
                frame = self.compute_thread.wait_frame()
                if frame is None:
                    self.running = False
                    break

                pose, frame_id, times = frame
                self.prep_times.append(times["camera"])
                self.cache_times.append(times["cache"])
                self.compute_times.append(times["compute"])

                if frame_id % 1 == 0:
                    x, y, z = get_translation(pose)
                    roll, pitch, yaw = get_rotation_degrees(pose)
                    frame = {
                        "time_budget": conf.FRAME_DIV * 33,
                        "prep_time": times["camera"] / 1000000,
                        "cache_time": times["cache"] / 1000000,
                        "compute_time": times["compute"] / 1000000,
                        "prep_time_min": min(self.prep_times) / 1000000,
                        "prep_time_max": max(self.prep_times) / 1000000,
                        "prep_time_avg": mean(self.prep_times) / 1000000,
                        "cache_time_min": min(self.cache_times) / 1000000,
                        "cache_time_max": max(self.cache_times) / 1000000,
                        "cache_time_avg": mean(self.cache_times) / 1000000,
                        "compute_time_min": min(self.compute_times) / 1000000,
                        "compute_time_max": max(self.compute_times) / 1000000,
                        "compute_time_avg": mean(self.compute_times) / 1000000,
                        "x": x,
                        "y": y,
                        "z": z,
                        "roll": roll,
                        "pitch": pitch,
                        "yaw": yaw,
                    }

                    with self.condition:
                        self.cached_frame = frame

        finally:
            pass

    def get_frame(self) -> Optional[dict]:
        with self.condition:
            return self.cached_frame


camera = None
camera_thread = None
frame_saver_thread = None
prepare_frame_thread = None
compute_thread = None
odometry_saver_thread = None
camera_lock = threading.Lock()


stream_counter = 0
def stream_frames(image="amplitude"):
    global camera_lock
    global camera
    global camera_thread
    global frame_saver_thread
    global stream_counter

    print("Streaming video")

    with camera_lock:
        if camera is None or camera_thread is None:
            camera = TofCamera(frame_timeout=0)
            camera_thread = CameraThread(camera)
            camera.start()
            camera_thread.start()

        if frame_saver_thread is None:
            frame_saver_thread = FrameSaverThread(camera_thread, camera)
            frame_saver_thread.start()

        stream_counter += 1

    try:
        while True:
            frame = frame_saver_thread.get_frame()
            if frame is None:
                continue
            amplitude, depth = frame
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
                time.sleep(0.01)
    finally:
        stream_counter -= 1

        if stream_counter == 0:
            print("Video preview quit.")
            with camera_lock:
                frame_saver_thread.stop()
                frame_saver_thread.join()
                frame_saver_thread = None



odometry_counter = 0
def stream_odometry():
    global camera_lock
    global camera
    global camera_thread
    global prepare_frame_thread
    global compute_thread
    global odometry_saver_thread
    global odometry_counter

    print("Streaming odometry")

    with camera_lock:
        if camera is None or camera_thread is None:
            camera = TofCamera(frame_timeout=0)
            camera_thread = CameraThread(camera)
            camera.start()
            camera_thread.start()

        if prepare_frame_thread is None or compute_thread is None or odometry_saver_thread is None:
            odometry = IcpOdometry(camera.get_intrinsic_matrix())
            prepare_frame_thread = PrepareFrameThread(camera_thread, odometry)
            compute_thread = ComputeThread(prepare_frame_thread, odometry)
            odometry_saver_thread = OdometrySaverThread(compute_thread)

            prepare_frame_thread.start()
            compute_thread.start()
            odometry_saver_thread.start()

        odometry_counter += 1

    try:
        while True:
            frame = odometry_saver_thread.get_frame()
            if frame is None:
                continue
            yield f"data: {json.dumps(frame)}\n\n"

            time.sleep(0.01)
    finally:
        odometry_counter -= 1

        if odometry_counter == 0:
            print("Odometry quit.")
            with camera_lock:
                camera_thread.pipeline_active = False
                prepare_frame_thread.stop()
                prepare_frame_thread.join()
                camera_thread.pipeline_active = False
                compute_thread.stop()
                compute_thread.join()
                camera_thread.pipeline_active = False
                odometry_saver_thread.stop()
                odometry_saver_thread.join()
                camera_thread.pipeline_active = False
                prepare_frame_thread = None
                compute_thread = None
                odometry_saver_thread = None
