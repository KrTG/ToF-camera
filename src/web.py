from collections import deque
import enum
import json
from statistics import mean
import os
import glob
import pickle
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
    RECORDING = 3
    PLAYBACK = 4

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


class MockDepthData:
    __mock__ = True
    def __init__(self, depth_data, amplitude_data, confidence_data):
        self.depth_data = depth_data
        self.amplitude_data = amplitude_data
        self.confidence_data = confidence_data

class RecorderThread(PipelineThread):
    def __init__(self, camera_thread: PipelineThread, camera: TofCamera, filename=None):
        super().__init__()
        self.camera_thread = camera_thread
        self.camera = camera
        self.frames_saved = 0
        self.filename_base = filename
        self.is_recording = False
        self.f = None
        self.lock = threading.Lock()

    def start_recording(self):
        with self.lock:
            if not self.is_recording:
                if self.filename_base is None:
                    os.makedirs("out", exist_ok=True)
                    existing_files = glob.glob(os.path.join("out", "*.replay"))
                    max_num = 0
                    for f in existing_files:
                        basename = os.path.basename(f)
                        name, ext = os.path.splitext(basename)
                        try:
                            num = int(name)
                            if num > max_num:
                                max_num = num
                        except ValueError:
                            pass
                    filename = os.path.join("out", f"{max_num + 1}.replay")
                else:
                    filename = self.filename_base
                self.f = open(filename, "wb")
                self.frames_saved = 0
                self.is_recording = True

    def stop_recording(self):
        with self.lock:
            if self.is_recording:
                self.is_recording = False
                if self.f is not None:
                    self.f.close()
                    self.f = None

    def run(self):
        self.running = True
        try:
            while self.running:
                frame = self.camera_thread.wait_frame()
                if frame is None:
                    self.running = False
                    break

                raw_frame, extra_data = frame

                with self.lock:
                    if self.is_recording and self.f is not None:
                        # Save mock frame
                        mock_frame = MockDepthData(
                            raw_frame.depth_data,
                            raw_frame.amplitude_data,
                            raw_frame.confidence_data
                        )

                        save_extra_data = extra_data.copy()
                        if "SYS_STATUS" in save_extra_data:
                            del save_extra_data["SYS_STATUS"]

                        pickle.dump((mock_frame, save_extra_data), self.f)
                        self.frames_saved += 1

                self.camera.release_frame_raw(raw_frame)

                with self.condition:
                    self.frame = frame
                    self.condition.notify()
        finally:
            self.stop_recording()
            with self.condition:
                self.running = False
                self.condition.notify_all()


class PlayerThread(PipelineThread):
    def __init__(self, filename: str, delay: float = 0.0):
        super().__init__()
        self.filename = filename
        self.delay = delay

    def run(self):
        self.running = True
        try:
            with open(self.filename, "rb") as f:
                while self.running:
                    try:
                        frame = pickle.load(f)

                        with self.condition:
                            self.frame = frame
                            self.condition.notify()

                        if self.delay > 0:
                            time.sleep(self.delay)
                    except EOFError:
                        print("Player: Reached end of recording")
                        self.running = False
                        break
                    except Exception as e:
                        print(f"Player error: {e}")
                        self.running = False
                        break
        except FileNotFoundError:
            print(f"Player: Recording file {self.filename} not found.")
            self.running = False
        finally:
            with self.condition:
                self.running = False
                self.condition.notify_all()

class OdometrySaverThread(PipelineThread):
    def __init__(self, compute_thread: PipelineThread):
        super().__init__()
        self.compute_thread = compute_thread

        self.cached_frame = None
        self.prep_times = deque(maxlen=100)
        self.cache_times = deque(maxlen=100)
        self.compute_times = deque(maxlen=100)
        self.missed_frames = 0

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
                self.missed_frames += 1 if not times["compute_success"] else 0

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
                        "missed_frames": self.missed_frames,
                        "x": x,
                        "y": y,
                        "z": z,
                        "roll": roll,
                        "pitch": pitch,
                        "yaw": yaw,
                        "voltage": times.get("SYS_STATUS").voltage_battery / 1000 / 4 if "SYS_STATUS" in times else 0,
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

        self.running = True
        self.lock = threading.Lock()
        self.watchdog_timer = time.monotonic()

    def run(self):
        while True:
            time.sleep(1)
            with self.lock:
                if not self.running:
                    return
                if (time.monotonic() - self.watchdog_timer) > self.timeout:
                    print(f"No ping for {self.timeout} seconds. Shutting down the threads.")
                    with self.streamer.camera_lock:
                        if self.streamer.watchdog_thread == self:
                            self.streamer.cleanup()
                        return

    def ping(self):
        with self.lock:
            self.watchdog_timer = time.monotonic()

    def stop(self):
        with self.lock:
            self.running = False

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
        self.recorder_thread = None
        self.player_thread = None
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

    def start_recording(self):
        mav_connection = mav.get_connection()
        heartbeat = mav_connection.wait_heartbeat(timeout=3)
        if heartbeat is None:
            raise RuntimeError("Recording needs a mavlink connection.")
        else:
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

        self.recorder_thread = RecorderThread(self.camera_thread, self.camera)
        self.recorder_thread.start()

    def cleanup_recording(self):
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread.join()
        if self.mav_connection is not None:
            self.mav_connection.close()
        if self.camera is not None:
            self.camera.stop()
        if self.recorder_thread is not None:
            self.recorder_thread.stop()
            self.recorder_thread.join()

        self.camera_thread = None
        self.camera = None
        self.recorder_thread = None
        self.mav_connection = None

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

    def start_playback(self, filename, delay: float = 0.0):
        if self.watchdog_thread is None:
            self.watchdog_thread = WatchdogThread(self)
            self.watchdog_thread.start()

        self.camera = TofCamera()
        self.camera.started = True # Enable conversion methods without HW access

        self.player_thread = PlayerThread(filename, delay=delay)
        self.player_thread.start()

    def cleanup_playback(self):
        if self.player_thread is not None:
            self.player_thread.stop()
            self.player_thread.join()
        self.player_thread = None
        self.camera = None

    def cleanup(self):
        if self.algorithm == Algorithm.VIDEO:
            self.cleanup_video()
        elif self.algorithm == Algorithm.ODOMETRY:
            self.cleanup_odometry()
        elif self.algorithm == Algorithm.RECORDING:
            self.cleanup_recording()
        elif self.algorithm == Algorithm.PLAYBACK:
            self.cleanup_playback()

        self.algorithm = Algorithm.NONE

    def stream_frames(self, image="amplitude"):
        with self.camera_lock:
            if self.algorithm != Algorithm.VIDEO:
                self.cleanup()
                self.start_video()
                self.algorithm = Algorithm.VIDEO

            print(f"Streaming video")

        frame_saver_thread = self.frame_saver_thread
        while True:
            if frame_saver_thread is None or not frame_saver_thread.running:
                return
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
                if self.watchdog_thread is not None:
                    self.watchdog_thread.ping()
                time.sleep(0.1)

    def stream_odometry(self):
        with self.camera_lock:
            if self.algorithm != Algorithm.ODOMETRY:
                self.cleanup()
                self.start_odometry()
                self.algorithm = Algorithm.ODOMETRY
            print("Streaming odometry")

        odometry_saver_thread = self.odometry_saver_thread
        while True:
            if odometry_saver_thread is None or not odometry_saver_thread.running:
                return
            frame = odometry_saver_thread.get_frame()
            if frame is None:
                continue
            yield f"data: {json.dumps(frame)}\n\n"
            if self.watchdog_thread is not None:
                self.watchdog_thread.ping()
            time.sleep(0.1)

    def stream_recording(self):
        with self.camera_lock:
            if self.algorithm != Algorithm.RECORDING:
                self.cleanup()
                self.start_recording()
                self.algorithm = Algorithm.RECORDING
            print("Streaming recording events")

        recorder_thread = self.recorder_thread
        while True:
            if recorder_thread is None or not recorder_thread.running:
                return

            yield f"data: {json.dumps({'frames_saved': recorder_thread.frames_saved})}\n\n"
            if self.watchdog_thread is not None:
                self.watchdog_thread.ping()
            time.sleep(0.1)

    def stream_playback_frames(self, filename, image="amplitude"):
        # Ensure the filename is within out or test for safety
        if not (filename.startswith("out/") or filename.startswith("test/")):
             # If it doesn't start with out/ or test/, maybe it's just a filename in out/ (backward compatibility)
             filename = os.path.join("out", filename)
        with self.camera_lock:
            if self.algorithm != Algorithm.PLAYBACK or \
               self.player_thread is None or \
               not self.player_thread.running or \
               self.player_thread.filename != filename:
                self.cleanup()
                self.start_playback(filename, delay=0.1)
                self.algorithm = Algorithm.PLAYBACK

            print(f"Streaming playback: {filename}")

        while True:
            if self.player_thread is None or not self.player_thread.running or self.camera is None:
                return

            frame_data = self.player_thread.wait_frame()
            if frame_data is None:
                continue

            raw_frame, extra_data = frame_data

            # Convert raw frame to images
            amplitude, depth, mask, _ = self.camera.get_frame_rgbd(raw_frame)
            amplitude_img = self.camera.convert_grayscale(amplitude, mask)
            depth_img = self.camera.convert_rgb(depth, mask)

            if image == "amplitude":
                im = amplitude_img
            elif image == "depth":
                im = depth_img

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

streamer = Streamer()
