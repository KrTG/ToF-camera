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

                amplitude, depth, mask, _time = self.camera.get_frame_rgbd(frame)
                self.camera.release_frame_raw(frame)

                extra_data = {}
                extra_data["camera_time"] = _time

                if self.mav_state is not None:
                    extra_data["ROTATION"] = interpolate(att_pre, time_pre, time_frame, att_post, time_post)
                    extra_data["SYS_STATUS"] = self.mav_state.sys_status

                frame = (amplitude, depth, mask, extra_data)

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


class PrepareFrameThread(PipelineThread):
    def __init__(self, camera_thread: PipelineThread, odometry: IcpOdometry):
        super().__init__()
        self.camera_thread = camera_thread
        self.odometry = odometry

        self.frame_counter = 0

    def run(self):
        self.running = True
        try:
            while self.running:
                frame = self.camera_thread.wait_frame()
                if frame is None:
                    self.running = False
                    break

                amplitude, depth, mask, extra_data = frame
                odometry_frame, _time = self.odometry.prepare_frame(
                    amplitude, depth, mask, self.frame_counter
                )
                extra_data["cache_time"] = _time
                frame = (odometry_frame, extra_data)

                self.frame_counter += 1

                with self.condition:
                    if self.frame is not None:
                        print(
                            "Prepare: Next frame ready, while previous was not acquired!"
                        )
                        self.running = False
                        break

                    self.frame = frame
                    self.condition.notify()
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

                odometry_frame, extra_data = frame
                pose, _time, t_error, r_error, c_error = self.odometry.compute_frame(odometry_frame, extra_data.get("ROTATION"))
                extra_data["compute_time"] = _time
                extra_data["t_error"] = t_error
                extra_data["r_error"] = r_error
                extra_data["c_error"] = c_error
                frame = (pose, odometry_frame.ID, extra_data)

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


def get_translation(pose):
    return (pose[0, 3], pose[1, 3], pose[2, 3])


def get_rotation_degrees(pose):
    rot_matrix = pose[:3, :3]
    r = Rotation.from_matrix(rot_matrix)
    yaw, pitch, roll = r.as_euler('zyx', degrees=True)

    return roll, pitch, yaw

def get_translation_from_vision_frame(pose):
    return (-pose[2, 3], -pose[0, 3], -pose[1, 3])


def get_rotation_from_vision_frame(pose):
    yaw = -math.atan2(-pose[2, 0], math.sqrt(pose[2, 1] ** 2 + pose[2, 2] ** 2))
    roll = -math.atan2(pose[1, 0], pose[0, 0])
    pitch = -math.atan2(pose[2, 1], pose[2, 2])
    return roll, pitch, yaw


def main():
    camera = TofCamera(frame_timeout=0)
    camera.start()
    odometry = IcpOdometry(camera.get_intrinsic_matrix())
    camera_thread = CameraThread(camera)
    prepare_frame_thread = PrepareFrameThread(camera_thread, odometry)
    compute_thread = ComputeThread(prepare_frame_thread, odometry)

    camera_thread.start()
    prepare_frame_thread.start()
    compute_thread.start()
    try:
        while True:
            frame = compute_thread.wait_frame()
            if frame is None:
                break
            global_pose, frame_id, times = frame
            prep_time = times["camera_time"]
            cache_time = times["cache_time"]
            compute_time = times["compute_time"]

            if (frame_id + 1) % 50 == 0:
                print(f"Time budget: {33 * conf.FRAME_DIV} ms")
                print(f"Pre-processing time: {prep_time / 1000000} ms")
                print(f"Prepare frame time: {cache_time / 1000000} ms")
                print(f"Compute odometry time: {compute_time / 1000000} ms")

                x, y, z = get_translation(global_pose)
                print(f"X (forwards/backwards): {x}")
                print(f"Y (right/left): {y}")
                print(f"Z (down/up): {z}")
                roll, pitch, yaw = get_rotation_degrees(global_pose)
                print(f"Roll: {roll}")
                print(f"Pitch: {pitch}")
                print(f"Yaw: {yaw}")

    finally:
        camera_thread.running = False
        prepare_frame_thread.running = False
        compute_thread.running = False
        camera_thread.join()
        prepare_frame_thread.join()
        compute_thread.join()


if __name__ == "__main__":
    main()
