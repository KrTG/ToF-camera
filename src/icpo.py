import time

import cv2
import numpy as np

from src.profileit import profileit
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
        gradient_magnitudes=conf.ICPO_GRADIENT_MAGNITUDES
    ):
        self.icpo = cv2.rgbd.RgbdICPOdometry.create(
            cameraMatrix=cam_matrix,
            minDepth=min_depth,
            maxDepth=max_depth,
            maxDepthDiff=max_depth_diff,
            maxPointsPart=max_points_part,
            iterCounts=iter_counts,
            minGradientMagnitudes=gradient_magnitudes,
            transformType=4 # Default
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

        self.anchor_odometry_frame = None
        self.previous_transform = np.eye(4, dtype=np.float64)
        self.global_pose = np.eye(4, dtype=np.float64)
        self.skipped_frames = 0

    def next_frame(self, amplitude, depth, mask, frame_id):
        _start_time = time.monotonic_ns()
        current_odometry_frame = cv2.rgbd.OdometryFrame.create(
            amplitude, depth, mask, None, frame_id
        )
        self.icpo.prepareFrameCache(
            current_odometry_frame, cv2.rgbd.ODOMETRY_FRAME_CACHE_ALL
        )
        _cache_time = time.monotonic_ns()

        if self.anchor_odometry_frame is not None:
            # Scale the initial transformation by the amount of lost frames
            init_rt = self.previous_transform
            if self.skipped_frames > 0:
                init_rt = np.linalg.matrix_power(init_rt, self.skipped_frames + 1)

            success, transform = self.icpo.compute2(
                self.anchor_odometry_frame,
                current_odometry_frame,
                initRt=init_rt
            )
            if success:
                self.global_pose @= transform
                self.anchor_odometry_frame = current_odometry_frame

                if self.skipped_frames == 0:
                    self.previous_transform = transform
                else:
                    # TODO: Average out the transform for initRt
                    self.skipped_frames = 0
            else:
                self.skipped_frames += 1
                print(f"Skipped frames: {self.skipped_frames}")
                if self.skipped_frames > 4:
                    print("Lost tracking. Re-set using linear prediction.")
                    jump = np.linalg.matrix_power(self.previous_transform, self.skipped_frames + 1)
                    self.global_pose @= jump
                    self.anchor_odometry_frame = current_odometry_frame
                    self.skipped_frames = 0
                    self.previous_transform = np.eye(4, dtype=np.float64)
        else:
            self.anchor_odometry_frame = current_odometry_frame

        _compute_time = time.monotonic_ns()

        return (
            self.global_pose,
            _compute_time - _start_time,
            _cache_time - _start_time,
            _compute_time - _cache_time,
        )
