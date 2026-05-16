import math
import os
import time
from threading import Condition, Thread
from typing import Optional, Tuple

from pymavlink import mavutil
from scipy.spatial.transform import Rotation

from src import conf, mav
from src.calc import interpolate
from src.icpo import IcpOdometry
from src.tof_camera import TofCamera


class PipelineThread(Thread):
    def __init__(self):
        super().__init__()

        self.running = False
        self.condition = Condition()
        self.frame = None
        self.pipeline_active = False

    def wait_frame(self) -> Optional[Tuple]:
        with self.condition:
            while self.frame is None and self.running:
                self.condition.wait()
            acquired_frame = self.frame
            self.frame = None
            self.pipeline_active = True
            return acquired_frame

    def stop(self):
        with self.condition:
            self.running = False
            self.condition.notify_all()


class CameraThread(PipelineThread):
    def __init__(self, camera: TofCamera, framerate_divisor: int = conf.FRAME_DIV, mav_connection: mavutil.mavserial | None = None):
        super().__init__()
        self.camera = camera
        self.divisor = framerate_divisor
        self.mav_state = None
        if mav_connection:
            self.mav_state = mav.StateMonitor(mav_connection, async_messages=["SYS_STATUS"], sync_messages=["ATTITUDE_QUATERNION"])

        self.frame_counter = 0

    def run(self):
        self.running = True
        try:
            while self.running:
                if self.mav_state is not None:
                    self.mav_state.update_state()
                    time_pre = self.mav_state.times["ATTITUDE_QUATERNION"]
                    att_pre = self.mav_state.current_state["ATTITUDE_QUATERNION"]

                frame = self.camera.get_frame_raw()
                time_frame = time.perf_counter()
                if frame is None:
                    time.sleep(0.0001)
                    continue

                self.frame_counter += 1
                if self.frame_counter % self.divisor != 0:
                    self.camera.release_frame_raw(frame)
                    continue

                if self.mav_state is not None:
                    self.mav_state.update_state()
                    time_post = self.mav_state.times["ATTITUDE_QUATERNION"]
                    att_post = self.mav_state.current_state["ATTITUDE_QUATERNION"]

                extra_data = {}
                if self.mav_state is not None:
                    extra_data["ROTATION"] = interpolate(att_pre, time_pre, time_frame, att_post, time_post)
                    extra_data["SYS_STATUS"] = self.mav_state.sys_status

                frame = (frame, extra_data)

                # We should process this frame
                with self.condition:
                    if self.pipeline_active:
                        if self.frame is not None:
                            print(
                                "Camera: Next frame ready, while previous was not acquired!"
                            )
                            self.running = False
                            break
                    self.frame = frame
                    self.condition.notify()
        finally:
            with self.condition:
                self.running = False
                self.condition.notify_all()

class PreprocessFrameThread(PipelineThread):
    def __init__(self, camera_thread: PipelineThread, camera: TofCamera):
        super().__init__()
        self.camera_thread = camera_thread
        self.camera = camera

    def run(self):
        self.running = True
        try:
            while self.running:

                frame = self.camera_thread.wait_frame()
                if frame is None:
                    self.running = False
                    break

                raw_frame, extra_data = frame

                amplitude, depth, mask, _time = self.camera.get_frame_rgbd(raw_frame)
                self.camera.release_frame_raw(raw_frame)

                extra_data["preprocess_time"] = _time
                frame = (amplitude, depth, mask, extra_data)

                # We should process this frame
                with self.condition:
                    if self.pipeline_active:
                        if self.frame is not None:
                            print(
                                "Preprocessor: Next frame ready, while previous was not acquired!"
                            )
                            self.running = False
                            break
                    self.frame = frame
                    self.condition.notify()
        finally:
            with self.condition:
                self.running = False
                self.condition.notify_all()


class PrepareCacheThread(PipelineThread):
    def __init__(self, camera_thread: PipelineThread, odometry: IcpOdometry):
        super().__init__()
        self.camera_thread = camera_thread
        self.odometry = odometry

        self.frame_counter = 0
        self.anchor_frame = None
        self.anchor_calculation_time = 0

    def run(self):
        self.running = True
        try:
            while self.running:
                frame = self.camera_thread.wait_frame()
                if frame is None:
                    self.running = False
                    break

                amplitude, depth, mask, extra_data = frame
                warped_frame, _time = self.odometry.prepare_warped_frame(
                    amplitude, depth, mask, self.frame_counter, extra_data["ROTATION"]
                )
                extra_data["cache_time"] = self.anchor_calculation_time + _time
                if self.anchor_frame is not None:
                    frame = (self.anchor_frame, warped_frame, extra_data)

                    with self.condition:
                        if self.frame is not None:
                            print(
                                "Prepare: Next frame ready, while previous was not acquired!"
                            )
                            self.running = False
                            break

                        self.frame = frame
                        self.condition.notify()
                self.anchor_frame, self.anchor_calculation_time = self.odometry.prepare_regular_frame(
                    amplitude, depth, mask, self.frame_counter
                )
                self.frame_counter += 1
        finally:
            with self.condition:
                self.running = False
                self.condition.notify_all()


class ComputeThread(PipelineThread):
    def __init__(self, prepare_frame_thread: PipelineThread, odometry: IcpOdometry):
        super().__init__()
        self.prepare_thread = prepare_frame_thread
        self.odometry = odometry

        self.frame = None

    def run(self):
        self.running = True
        try:
            while self.running:
                frame = self.prepare_thread.wait_frame()
                if frame is None:
                    self.running = False
                    break

                anchor_frame, warped_frame, extra_data = frame
                pose, success, _time = self.odometry.compute_frame(
                    anchor_frame, warped_frame, extra_data["ROTATION"]
                )
                extra_data["compute_time"] = _time
                extra_data["compute_success"] = success
                extra_data["id"] = warped_frame.ID
                frame = (pose, extra_data)

                if os.path.isfile("/tmp/reset"):
                    os.remove("/tmp/reset")
                    self.odometry.reset_position()
                    print("Position reset by user.")

                with self.condition:
                    if self.frame is not None:
                        print(
                            "Compute: Next frame ready, while previous was not acquired!"
                        )
                        self.running = False
                        break
                    self.frame = frame
                    self.condition.notify()
        finally:
            with self.condition:
                self.running = False
                self.condition.notify_all()



class OutputMavlinkThread(PipelineThread):
    def __init__(self, compute_thread: PipelineThread, mav_connection: mavutil.mavserial):
        super().__init__()
        self.compute_thread = compute_thread
        self.commander = mav.Commander(mav_connection)

    def run(self):
        self.running = True
        try:
            while self.running:
                # Wait for a new frame from the compute thread
                frame_data = self.compute_thread.wait_frame()
                if frame_data is None:
                    self.running = False
                    break

                pose, extra_data = frame_data
                success = extra_data["compute_success"]
                assert isinstance(success, bool)

                x, y, z = get_translation(pose)
                qw, qx, qy, qz = get_rotation_quaternion(pose)

                self.commander.odometry(x, y, z, qw, qx, qy, qz)


        finally:
            with self.condition:
                self.running = False
                self.condition.notify_all()


def get_translation(pose):
    return (pose[0, 3], pose[1, 3], pose[2, 3])


def get_rotation_degrees(pose):
    rot_matrix = pose[:3, :3]
    r = Rotation.from_matrix(rot_matrix)
    yaw, pitch, roll = r.as_euler('zyx', degrees=True)

    return roll, pitch, yaw

def get_rotation_quaternion(pose):
    rot_matrix = pose[:3, :3]
    r = Rotation.from_matrix(rot_matrix)
    w, x, y, z = r.as_quat(canonical=True, scalar_first=True)

    return w, x, y, z
