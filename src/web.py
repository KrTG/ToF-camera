from collections import deque
import enum
import json
from statistics import mean
import threading
import time
from typing import Optional, Tuple

import cv2
from pymavlink import mavutil

from src import mav
from src.estimator import (CameraThread, ComputeThread, PipelineThread,
                       PrepareCacheThread, PreprocessFrameThread, get_rotation_degrees,
                       get_translation)
from src import conf
from src.icpo import IcpOdometry
from src.tof_camera import TofCamera


class Algorithm(enum.Enum):
    NONE = 0
    ODOMETRY = 1
    VIDEO = 2

class FrameSaverThread(PipelineThread):
    def __init__(self, camera_thread: PipelineThread, camera: TofCamera):
        super().__init__()
        self.camera_thread = camera_thread
        self.camera = camera
        self.cached_frame = None

    def run(self):
        self.running = True
        try:
            while self.running:
                frame = self.camera_thread.wait_frame()
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
        self.t_errors = deque(maxlen=100)
        self.r_errors = deque(maxlen=100)

    def run(self):
        self.running = True
        try:
            while self.running:
                frame = self.compute_thread.wait_frame()
                if frame is None:
                    self.running = False
                    break

                pose, frame_id, times = frame
                self.prep_times.append(times["preprocess_time"])
                self.cache_times.append(times["cache_time"])
                self.compute_times.append(times["compute_time"])
                self.t_errors.append(times["t_error"])
                self.r_errors.append(times["r_error"])

                if frame_id % 1 == 0:
                    x, y, z = get_translation(pose)
                    roll, pitch, yaw = get_rotation_degrees(pose)
                    frame = {
                        "time_budget": conf.FRAME_DIV * 33,
                        "prep_time": times["preprocess_time"] / 1000000,
                        "cache_time": times["cache_time"] / 1000000,
                        "compute_time": times["compute_time"] / 1000000,
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
                        "voltage": times.get["SYS_STATUS"].voltage_battery / 1000 / 4 if "SYS_STATUS" in times else 0,
                        "t_error": times["t_error"],
                        "r_error": times["r_error"],
                        "t_error_min": min(self.t_errors),
                        "t_error_max": max(self.t_errors),
                        "t_error_avg": mean(self.t_errors),
                        "r_error_min": min(self.r_errors),
                        "r_error_max": max(self.r_errors),
                        "r_error_avg": mean(self.r_errors),
                    }

                    with self.condition:
                        self.cached_frame = frame

        finally:
            pass

    def get_frame(self) -> Optional[dict]:
        with self.condition:
            return self.cached_frame


class WatchdogThread(threading.Thread):
    def __init__(self, streamer, timeout=5):
        super().__init__()
        self.streamer = streamer
        self.timeout = timeout

        self.running = False
        self.lock = threading.Lock()
        self.watchdog_timer = time.monotonic()

    def run(self):
        while True:
            with self.lock:
                if (time.monotonic() - self.watchdog_timer) > self.timeout:
                    print(f"No ping for {self.timeout} seconds. Shutting down the threads.")
                    with self.streamer.camera_lock:
                        self.running = False
                        self.streamer.cleanup()
                        return
            time.sleep(1)

    def ping(self):
        with self.lock:
            self.watchdog_timer = time.monotonic()

class Streamer:
    def __init__(self):
        self.camera = None
        self.mav_connection = None
        self.camera_thread = None
        self.preprocess_frame_thread = None
        self.frame_saver_thread = None
        self.prepare_cache_thread = None
        self.compute_thread = None
        self.odometry_saver_thread = None
        self.watchdog_thread = None
        self.camera_lock = threading.Lock()

        self.algorithm = Algorithm.NONE

    def start_video(self):
        if self.watchdog_thread is None:
            self.watchdog_thread = WatchdogThread(self)
            self.watchdog_thread.start()

        self.camera = TofCamera(frame_timeout=0)
        self.camera_thread = CameraThread(self.camera)
        self.camera.start()
        self.camera_thread.start()
        self.preprocess_frame_thread = PreprocessFrameThread(self.camera_thread, self.camera)
        self.preprocess_frame_thread.start()
        self.frame_saver_thread = FrameSaverThread(self.preprocess_frame_thread, self.camera)
        self.frame_saver_thread.start()

    def cleanup_video(self):
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread.join()
        if self.camera is not None:
            self.camera.stop()
        if self.preprocess_frame_thread is not None:
            self.preprocess_frame_thread.stop()
            self.preprocess_frame_thread.join()
        if self.frame_saver_thread is not None:
            self.frame_saver_thread.stop()
            self.frame_saver_thread.join()
        self.frame_saver_thread = None
        self.camera_thread = None
        self.camera = None

    def start_odometry(self):
        mav_connection = mav.get_connection()
        heartbeat = mav_connection.wait_heartbeat(timeout=3)
        if heartbeat is not None:
            self.mav_connection = mav_connection
            mav.Commander(mav_connection).set_message_interval(
                mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_QUATERNION, 6500
            )  # 150 FPS

        if self.watchdog_thread is None:
            self.watchdog_thread = WatchdogThread(self)
            self.watchdog_thread.start()

        self.camera = TofCamera(frame_timeout=0)
        self.camera_thread = CameraThread(self.camera, mav_connection=self.mav_connection)
        self.camera.start()
        self.camera_thread.start()
        self.preprocess_frame_thread = PreprocessFrameThread(self.camera_thread, self.camera)
        self.preprocess_frame_thread.start()
        self.odometry = IcpOdometry(self.camera.get_intrinsic_matrix())
        self.prepare_cache_thread = PrepareCacheThread(self.preprocess_frame_thread, self.odometry)
        self.compute_thread = ComputeThread(self.prepare_cache_thread, self.odometry)
        self.odometry_saver_thread = OdometrySaverThread(self.compute_thread)

        self.prepare_cache_thread.start()
        self.compute_thread.start()
        self.odometry_saver_thread.start()

    def cleanup_odometry(self):
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread.join()
        if self.mav_connection is not None:
            self.mav_connection.close()
        if self.camera is not None:
            self.camera.stop()
        if self.preprocess_frame_thread is not None:
            self.preprocess_frame_thread.stop()
            self.preprocess_frame_thread.join()
        if self.prepare_cache_thread is not None:
            self.prepare_cache_thread.stop()
            self.prepare_cache_thread.join()
        if self.compute_thread is not None:
            self.compute_thread.stop()
            self.compute_thread.join()
        if self.odometry_saver_thread is not None:
            self.odometry_saver_thread.stop()
            self.odometry_saver_thread.join()

        self.camera_thread = None
        self.camera = None
        self.prepare_cache_thread = None
        self.compute_thread = None
        self.odometry_saver_thread = None

    def cleanup(self):
        if self.algorithm == Algorithm.VIDEO:
            self.cleanup_video()
        elif self.algorithm == Algorithm.ODOMETRY:
            self.cleanup_odometry()

        self.watchdog_thread = None
        self.algorithm = Algorithm.NONE

    def stream_frames(self, image="amplitude"):
        with self.camera_lock:
            if self.algorithm == Algorithm.ODOMETRY:
                self.cleanup_odometry()
                self.algorithm = Algorithm.NONE

            if self.algorithm == Algorithm.NONE:
                self.start_video()
                self.algorithm = Algorithm.VIDEO

            print(f"Streaming video")

        while True:
            if self.frame_saver_thread is None:
                return
            frame = self.frame_saver_thread.get_frame()
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
                if self.watchdog_thread is not None:
                    self.watchdog_thread.ping()
                time.sleep(0.1)

    def stream_odometry(self):
        with self.camera_lock:
            if self.algorithm == Algorithm.VIDEO:
                self.cleanup_video()
                self.algorithm = Algorithm.NONE

            if self.algorithm == Algorithm.NONE:
                self.start_odometry()
                self.algorithm = Algorithm.ODOMETRY
            print("Streaming odometry")

        while True:
            if self.odometry_saver_thread is None:
                return
            frame = self.odometry_saver_thread.get_frame()
            if frame is None:
                continue
            yield f"data: {json.dumps(frame)}\n\n"
            if self.watchdog_thread is not None:
                self.watchdog_thread.ping()
            time.sleep(0.1)

streamer = Streamer()
