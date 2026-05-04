import time

import cv2
import numpy as np
from cv2.rgbd import OdometryFrame
from scipy.linalg import expm, logm
from scipy.spatial.transform import Rotation

from src import conf


class IcpOdometry:
    def __init__(
        self,
        cam_matrix,
        min_depth=conf.ICPO_MIN_DEPTH,
        max_depth=conf.ICPO_MAX_DEPTH,
        max_depth_diff=conf.ICPO_MAX_DEPTH_DIFF,
        max_points_part=conf.ICPO_MAX_POINTS_PART,
        iter_counts=conf.ICPO_ITER_COUNTS,
        gradient_magnitudes=conf.ICPO_GRADIENT_MAGNITUDES,
    ):
        self.icpo = cv2.rgbd.RgbdICPOdometry.create(
            cameraMatrix=cam_matrix,
            minDepth=min_depth,
            maxDepth=max_depth,
            maxDepthDiff=max_depth_diff,
            maxPointsPart=max_points_part,
            iterCounts=iter_counts,
            minGradientMagnitudes=gradient_magnitudes,
            transformType=4,  # Default
        )
        print("----ICPO SETTINGS----")
        print(f"Camera matrix: {self.icpo.getCameraMatrix()}")
        print(f"Min depth: {self.icpo.getMinDepth()}")
        print(f"Max depth: {self.icpo.getMaxDepth()}")
        print(f"Max depth diff: {self.icpo.getMaxDepthDiff()}")
        print(f"Max points part: {self.icpo.getMaxPointsPart()}")
        print(f"Iter counts: {self.icpo.getIterationCounts()}")
        print(f"Min gradient magnitudes: {self.icpo.getMinGradientMagnitudes()}")
        print(f"Transform type: {self.icpo.getTransformType()}")
        print("--------")

        self.anchor_odometry_frame: OdometryFrame | None = None
        self.anchor_attitude: Rotation | None = None
        self.previous_transform: np.ndarray = np.eye(
            4, dtype=np.float64
        )  # 4x4 transform matrix
        self.global_pose: np.ndarray = np.eye(
            4, dtype=np.float64
        )  # 4x4 transform matrix
        self.skipped_frames: int = 0

    def prepare_frame(self, amplitude, depth, mask, frame_id):
        _start_time = time.monotonic_ns()
        current_odometry_frame = cv2.rgbd.OdometryFrame.create(
            amplitude, depth, mask, None, frame_id
        )
        self.icpo.prepareFrameCache(
            current_odometry_frame, cv2.rgbd.ODOMETRY_FRAME_CACHE_ALL
        )

        return current_odometry_frame, time.monotonic_ns() - _start_time

    def compute_frame(self, odometry_frame, attitude_quaternion_msg=None):
        _start_time = time.monotonic_ns()

        attitude = None
        if attitude_quaternion_msg is not None:
            att = attitude_quaternion_msg
            attitude = np.array([att.q1, att.q2, att.q3, att.q4])
            attitude = Rotation.from_quat(attitude)

        if self.anchor_odometry_frame is not None:
            # Scale the initial transformation by the amount of lost frames
            init_rt = self.previous_transform
            if self.skipped_frames > 0:
                init_rt = np.linalg.matrix_power(init_rt, self.skipped_frames + 1)

            if attitude is not None and self.anchor_attitude is not None:
                att_delta = attitude * self.anchor_attitude.inv()
                init_rt[:3, :3] = att_delta.as_matrix()

            success, transform = self.icpo.compute2(
                self.anchor_odometry_frame, odometry_frame, initRt=init_rt
            )
            if success:
                self.global_pose @= transform

                if attitude is not None:
                    self.global_pose[:3, :3] = attitude.as_matrix()

                self.anchor_odometry_frame = odometry_frame
                if attitude is not None:
                    self.anchor_attitude = attitude

                if self.skipped_frames == 0:
                    self.previous_transform = transform
                else:
                    self.previous_transform = expm(logm(transform) / (self.skipped_frames + 1)).real  # type: ignore
                    self.skipped_frames = 0
            else:
                self.skipped_frames += 1
                print(f"Skipped frames: {self.skipped_frames}")
                if self.skipped_frames > 2:
                    print("Lost tracking. Re-set using linear prediction.")
                    # Apply the 'guess' as the real prediction since we lost tracking
                    # and it's the best compromise
                    self.global_pose @= init_rt
                    self.anchor_odometry_frame = odometry_frame
                    if attitude is not None:
                        self.anchor_attitude = attitude
                    self.skipped_frames = 0
                    self.previous_transform = np.eye(4, dtype=np.float64)
        else:
            self.anchor_odometry_frame = odometry_frame
            if attitude is not None:
                self.anchor_attitude = attitude

        return self.global_pose, time.monotonic_ns() - _start_time

    def next_frame(self, amplitude, depth, mask, frame_id):
        _start_time = time.monotonic_ns()
        current_odometry_frame, _prep_time = self.prepare_frame(
            amplitude, depth, mask, frame_id
        )
        pose, _compute_time = self.compute_frame(current_odometry_frame)
        _total_time = time.monotonic_ns() - _start_time

        return (pose, _total_time, _prep_time, _compute_time)

    def reset_position(self):
        self.global_pose = np.eye(4, dtype=np.float64)
