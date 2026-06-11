import os
import time
from threading import Condition, Thread
from typing import Optional, Tuple

import numpy as np
from pymavlink import mavutil
from scipy.spatial.transform import Rotation

from src import conf, mav
from src.calc import interpolate_attitude, interpolate_acceleration
from src.icpo import IcpOdometry
from src.tof_camera import TofCamera
from src.log import get_logger
from src import led


class PipelineThread(Thread):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(self.__class__.__name__)
        self.running = False
        self.condition = Condition()
        self.frame = None

    def wait_frame(self) -> Optional[Tuple]:
        with self.condition:
            while self.frame is None and self.running:
                self.condition.wait()
            acquired_frame = self.frame
            self.frame = None
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
            self.mav_state = mav.StateMonitor(mav_connection, async_messages=["SYS_STATUS"], sync_messages=["ATTITUDE_QUATERNION", "HIGHRES_IMU"])
            for _ in range(30):
                self.mav_state.timesync()
        self.frame_counter = 0

    def run(self):
        self.running = True
        try:
            while self.running:
                if self.mav_state is not None:
                    self.mav_state.update_state()
                    if not self.mav_state.is_initialized():
                        continue
                    att_pre = self.mav_state.current_state["ATTITUDE_QUATERNION"]
                    pos_pre = self.mav_state.current_state["HIGHRES_IMU"]

                frame = self.camera.get_frame_raw()
                if self.mav_state is not None:
                    time_ns = self.mav_state.time_ns()
                else:
                    time_ns = time.monotonic_ns()

                if frame is None:
                    continue

                self.frame_counter += 1
                if self.frame_counter % self.divisor != 0:
                    self.camera.release_frame_raw(frame)
                    continue

                if self.mav_state is not None:
                    self.mav_state.update_state()
                    att_post = self.mav_state.current_state["ATTITUDE_QUATERNION"]
                    pos_post = self.mav_state.current_state["HIGHRES_IMU"]

                extra_data = {}
                if self.mav_state is not None:
                    extra_data["SYS_STATUS"] = self.mav_state.sys_status
                    extra_data["rotation"] = interpolate_attitude(att_pre, att_post, time_ns / 1_000_000)
                    extra_data["acceleration"] = interpolate_acceleration(pos_pre, pos_post, time_ns / 1_000)
                extra_data["frame_timestamp"] = time_ns
                extra_data["ID"] = self.frame_counter // self.divisor
                frame = (frame, extra_data)

                # We should process this frame
                with self.condition:
                    if self.frame is not None:
                        self.logger.warning(
                            "Camera: Next frame ready, while previous was not acquired!"
                        )
                    self.frame = frame
                    self.condition.notify()
                if self.mav_state is not None:
                    self.mav_state.timesync()
        except Exception as e:
            self.logger.exception("CameraThread terminated due to an unhandled exception.")
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

                with self.condition:
                    if self.frame is not None:
                        self.logger.warning(
                            "Preprocessor: Next frame ready, while previous was not acquired!"
                        )
                    self.frame = frame
                    self.condition.notify()
        except Exception as e:
            self.logger.exception("PreprocessFrameThread terminated due to an unhandled exception.")
        finally:
            with self.condition:
                self.running = False
                self.condition.notify_all()


class PrepareCacheThread(PipelineThread):
    def __init__(self, camera_thread: PipelineThread, odometry: IcpOdometry):
        super().__init__()
        self.camera_thread = camera_thread
        self.odometry = odometry

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
                    amplitude, depth, mask, extra_data["ID"], extra_data["rotation"]
                )
                quality = self.odometry.integrate_quality_mask(mask)
                extra_data["cache_time"] = self.anchor_calculation_time + _time
                extra_data["quality"] = quality
                if self.anchor_frame is not None:
                    frame = (self.anchor_frame, warped_frame, extra_data)

                    with self.condition:
                        if self.frame is not None:
                            self.logger.warning(
                                "Prepare: Next frame ready, while previous was not acquired!"
                            )
                        self.frame = frame
                        self.condition.notify()
                self.anchor_frame, self.anchor_calculation_time = self.odometry.prepare_regular_frame(
                    amplitude, depth, mask, extra_data["ID"]
                )
        except Exception as e:
            self.logger.exception("PrepareCacheThread terminated due to an unhandled exception.")
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
                    anchor_frame, warped_frame, extra_data["rotation"], extra_data["acceleration"]
                )
                quality = self.odometry.integrate_quality_success(success)
                extra_data["compute_time"] = _time
                extra_data["compute_success"] = success
                extra_data["id"] = warped_frame.ID
                extra_data["quality"] = quality
                frame = (pose, extra_data)

                if os.path.isfile("/tmp/reset"):
                    os.remove("/tmp/reset")
                    self.odometry.reset_position()
                    self.logger.info("Position reset by user.")

                with self.condition:
                    if self.frame is not None:
                        self.logger.warning(
                            "Compute: Next frame ready, while previous was not acquired!"
                        )
                    self.frame = frame
                    self.condition.notify()
        except Exception as e:
            self.logger.exception("ComputeThread terminated due to an unhandled exception.")
        finally:
            with self.condition:
                self.running = False
                self.condition.notify_all()


class OutputMavlinkThread(PipelineThread):
    def __init__(self, compute_thread: PipelineThread, mav_connection: mavutil.mavserial):
        super().__init__()
        self.compute_thread = compute_thread
        self.commander = mav.Commander(mav_connection)

        self.is_sending = False
        self.quality_range = (0.25, 0.35)
        self.reset_counter = 0

    def run(self):
        self.running = True
        try:
            blue_led = led.get_blue()
            green_led = led.get_green()
            blue_led.on()
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

                if extra_data["ID"] % conf.FPS == 0:
                    self.commander.send_heartbeat()

                # Stop sending when quality is very low
                # Re-start sending when quality gets average
                if self.is_sending:
                    if extra_data["quality"] < self.quality_range[0]:
                        self.logger.warning("Output: Stopping sending odometry. Quality too low!")
                        self.is_sending = False
                        blue_led.on()
                        green_led.off()
                else:
                    if extra_data["quality"] > self.quality_range[1]:
                        self.logger.info("Output: Starting sending odometry. Quality regained.")
                        self.reset_counter += 1
                        self.is_sending = True
                        green_led.on()
                        blue_led.off()

                # For now we do not use the quality field and control
                # sending ourselves as I don't know what does this
                # affect and how it works on the drone side.
                if self.is_sending:
                    self.commander.odometry(
                        x, y, z, qw, qx, qy, qz,
                        timestamp=extra_data["frame_timestamp"] // 1000,
                        reset_counter=self.reset_counter
                    )
        except Exception as e:
            self.logger.exception("OutputMavlinkThread terminated due to an unhandled exception.")
        finally:
            blue_led.off()
            blue_led.close()
            green_led.off()
            green_led.close()
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
