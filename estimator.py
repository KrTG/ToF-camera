import math
import os
from threading import Condition, Lock, Thread

from src.icpo import IcpOdometry
from src.tof_camera import TofCamera


camera = TofCamera(scale=0.7, frame_timeout=6000)
camera.start()

odometry = IcpOdometry(camera.get_intrinsic_matrix())

last_frame_time = None
frame_times = []
frame_number = 0


class CameraThread(Thread):
    def __init__(self, camera: TofCamera, framerate_divisor: int = 1):
        super().__init__()
        self.camera = camera
        self.divisor = framerate_divisor

        self.running = False
        self.condition = Condition()
        self.frame = None
        self.frame_counter = 0

    def run(self):
        self.running = True
        try:
            while self.running:
                frame = self.camera.get_rgbd()
                if frame is None:
                    print("Camera: camera timed out.")
                    self.running = False
                    break

                amplitude, depth, mask, _time = frame
                frame = (amplitude, depth, mask, {"camera": _time})

                self.frame_counter += 1

                if self.frame_counter % self.divisor == 0:
                    # We should process this frame
                    with self.condition:
                        if self.frame is not None:
                            print("Camera: Next frame ready, while previous was not acquired!")
                            self.running = False
                            break
                        self.frame = frame
                        self.condition.notify()
        finally:
            with self.condition:
                 self.running = False
                 self.condition.notify_all()

    def get_frame(self):
        with self.condition:
            while self.frame is None and self.running:
                self.condition.wait()
            acquired_frame = self.frame
            self.frame = None
            return acquired_frame

class PrepareFrameThread(Thread):
    def __init__(self, camera_thread: CameraThread, odometry: IcpOdometry):
        super().__init__()
        self.camera_thread = camera_thread
        self.odometry = odometry

        self.running = False
        self.condition = Condition()
        self.frame = None
        self.frame_counter = 0

    def run(self):
        self.running = True
        try:
            while self.running:
                frame = self.camera_thread.get_frame()
                if frame is None:
                    self.running = False
                    break

                amplitude, depth, mask, times = frame
                odometry_frame, _time = self.odometry.prepare_frame(amplitude, depth, mask, self.frame_counter)
                times["prepare_frame"] = _time
                frame = (odometry_frame, times)

                self.frame_counter += 1

                with self.condition:
                    if self.frame is not None:
                        print("Prepare: Next frame ready, while previous was not acquired!")
                        self.running = False
                        break

                    self.frame = frame
                    self.condition.notify()
        finally:
            with self.condition:
                 self.running = False
                 self.condition.notify_all()

    def get_frame(self):
        with self.condition:
            while self.frame is None and self.running:
                self.condition.wait()
            acquired_frame = self.frame
            self.frame = None
            return acquired_frame

class ComputeThread(Thread):
    def __init__(self, prepare_frame_thread: PrepareFrameThread, odometry: IcpOdometry):
        super().__init__()
        self.prepare_thread = prepare_frame_thread
        self.odometry = odometry

        self.running = False
        self.condition = Condition()
        self.frame = None

    def run(self):
        self.running = True
        try:
            while self.running:
                frame = self.prepare_thread.get_frame()
                if frame is None:
                    self.running = False
                    break

                odometry_frame, times = frame
                pose, _time = self.odometry.compute_frame(odometry_frame)
                times["compute_frame"] = _time
                frame = (pose, times)

                with self.condition:
                    if self.frame is not None:
                        print("Compute: Next frame ready, while previous was not acquired!")
                        self.running = False
                        break
                    self.frame = frame
                    self.condition.notify()
        finally:
            with self.condition:
                 self.running = False
                 self.condition.notify_all()

    def get_frame(self):
        with self.condition:
            while self.frame is None and self.running:
                self.condition.wait()
            acquired_frame = self.frame
            self.frame = None
            return acquired_frame


# Everything is inverted

def get_translation(pose):
    return (-pose[2, 3], -pose[0, 3], -pose[1, 3])


def get_rotation(pose):
    yaw = -math.atan2(-pose[2, 0], math.sqrt(pose[2, 1] ** 2 + pose[2, 2] ** 2))
    roll = -math.atan2(pose[1, 0], pose[0, 0])
    pitch = -math.atan2(pose[2, 1], pose[2, 2])
    return roll, pitch, yaw

def get_rotation_degrees(pose):
    r, p, y = get_rotation(pose)
    return math.degrees(r), math.degrees(p), math.degrees(y)


while True:
    amplitude, depth, mask, prep_time = camera.get_rgbd()

    if amplitude is not None and depth is not None and mask is not None:
        if os.path.isfile("/tmp/reset"):
            os.remove("/tmp/reset")
            odometry.reset_position()
            print("Position reset by user.")
        global_pose, odo_time, cache_time, compute_time = odometry.next_frame(
            amplitude, depth, mask, frame_number
        )

        frame_number += 1
    else:
        raise RuntimeError(
            f"Frames should not be getting dropped.{amplitude}, {depth}, {mask}"
        )

    if (frame_number + 1) % 100 == 0:
        print("Time budget: 33 ms")
        print(f"Pre-processing time: {prep_time / 1000000} ms")
        print(
            f"Processing time: {odo_time / 1000000} ms (cache: {cache_time / 1000000}, compute: {compute_time / 1000000})"
        )
        x, y, z = get_translation(global_pose)
        print(f"X (forwards/backwards): {x}")
        print(f"Y (right/left): {y}")
        print(f"Z (down/up): {z}")
        roll, pitch, yaw = get_rotation_degrees(global_pose)
        print(f"Roll: {roll}")
        print(f"Pitch: {pitch}")
        print(f"Yaw: {yaw}")
