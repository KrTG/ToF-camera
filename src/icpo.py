import time

import cv2
from cv2.typing import MatLike
import numpy as np
from cv2.rgbd import OdometryFrame
from scipy.linalg import expm, logm
from scipy.spatial.transform import Rotation

from src import conf


def fast_inversion(transform):
    R_inv = transform[:3, :3].T
    t_inv = -R_inv @ transform[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = R_inv
    inv[:3, 3] = t_inv

    return inv


def fractional_se3_power(transform: np.ndarray, scale: float) -> np.ndarray:
    scaled_transform = np.eye(4, dtype=np.float64)

    rot_vec = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    scaled_transform[:3, :3] = Rotation.from_rotvec(rot_vec * scale).as_matrix()
    scaled_transform[:3, 3] = transform[:3, 3] * scale

    return scaled_transform


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
        self.cam_matrix = cam_matrix.astype(np.float32)
        self.icpo = cv2.rgbd.RgbdICPOdometry.create(
            cameraMatrix=cam_matrix,
            minDepth=min_depth,
            maxDepth=max_depth,
            maxDepthDiff=max_depth_diff,
            maxPointsPart=max_points_part,
            iterCounts=iter_counts,
            minGradientMagnitudes=gradient_magnitudes,
            transformType=cv2.rgbd.ODOMETRY_TRANSLATION,
        )

        if conf.DEBUG:
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

        self.anchor_attitude: Rotation | None = None
        self.previous_transform: np.ndarray = np.eye(4, dtype=np.float64)
        self.camera_mount_euler = (0, -90, 0) # Camera rotation in the FRD frame - facing down

        self.camera_mount_rotation = Rotation.from_euler('xyz', self.camera_mount_euler, degrees=True)
        self.frd_to_rdf_rotation = Rotation.from_matrix([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])

        self.rdf_to_frd_rotation = Rotation.from_matrix([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])

        self.frd_to_rdf_transform = np.eye(4)
        self.frd_to_rdf_transform[:3, :3] = self.frd_to_rdf_rotation.as_matrix()

        self.rdf_to_frd_transform = np.eye(4)
        self.rdf_to_frd_transform[:3, :3] = self.rdf_to_frd_rotation.as_matrix()

        self.camera_mount_transform = np.eye(4)
        self.camera_mount_transform[:3, :3] = self.camera_mount_rotation.as_matrix()

        # Cached ops
        self.frd_to_rdf_rotation_inv = self.rdf_to_frd_rotation
        self.final_transform = self.rdf_to_frd_transform.T @ self.camera_mount_transform.T

        self.reset_position()

    def prepare_warped_frame(
            self, amplitude: MatLike, depth: MatLike, mask: MatLike, frame_id: int, rotation: Rotation
        ):
        _start_time = time.monotonic_ns()
        # Current attitude in RDF frame
        attitude = rotation * self.camera_mount_rotation
        attitude = self.frd_to_rdf_rotation * attitude * self.frd_to_rdf_rotation_inv
        warped_frame = None
        if self.anchor_attitude is not None:
            relative_attitude = self.anchor_attitude.inv() * attitude

            T_warp = np.eye(4, dtype=np.float32)
            T_warp[:3, :3] = relative_attitude.as_matrix().astype(np.float32)

            K = self.cam_matrix
            distCoeff = np.array([0, 0, 0, 0])

            warped_amplitude = np.zeros_like(amplitude)
            warped_depth = np.zeros_like(depth)
            warped_mask = np.zeros_like(mask)

            cv2.rgbd.warpFrame(
                amplitude, depth, mask,
                T_warp, K, distCoeff,
                warped_amplitude, warped_depth, warped_mask
            )
            warped_frame = cv2.rgbd.OdometryFrame.create(
                warped_amplitude, warped_depth, warped_mask, None, frame_id
            )
            self.icpo.prepareFrameCache(warped_frame, cv2.rgbd.ODOMETRY_FRAME_CACHE_DST)
        self.anchor_attitude = attitude
        return warped_frame, time.monotonic_ns() - _start_time

    def prepare_regular_frame(
        self, amplitude: MatLike, depth: MatLike, mask: MatLike, frame_id: int
    ):
        _start_time = time.monotonic_ns()
        regular_frame = cv2.rgbd.OdometryFrame.create(
            amplitude, depth, mask, None, frame_id
        )
        self.icpo.prepareFrameCache(
            regular_frame, cv2.rgbd.ODOMETRY_FRAME_CACHE_SRC
        )
        return regular_frame, time.monotonic_ns() - _start_time

    def compute_frame(self, anchor_frame: OdometryFrame, warped_frame: OdometryFrame, rotation: Rotation):
        """
        @param anchor_frame: Unwarped previous frame (anchor)
        @param warped_frame: Current warped frame
        """
        assert (warped_frame.ID - anchor_frame.ID) == 1

        _start_time = time.monotonic_ns()

        init_rt = self.previous_transform.copy()

        # ICP is now translation-only (transformType=2)
        # Both frames are now aligned to the anchor's orientation
        success, transform = self.icpo.compute2(
            anchor_frame, warped_frame, initRt=init_rt
        )
        if success:
            self.global_pose @= fast_inversion(transform)
            attitude = rotation * self.camera_mount_rotation
            attitude = self.frd_to_rdf_rotation * attitude * self.frd_to_rdf_rotation_inv
            self.global_pose[:3, :3] = attitude.as_matrix()

            self.previous_transform = transform
        else:
            if conf.DEBUG:
                print("Lost tracking. Re-set using linear prediction.")
            # Apply the 'guess' as the real prediction since we lost tracking
            # and it's the best compromise
            self.global_pose @= fast_inversion(init_rt)

            # Either reset or set to init_rt - we choose to reset
            self.previous_transform = np.eye(4, dtype=np.float64)

        pose = self.rdf_to_frd_transform @ self.global_pose @ self.final_transform
        return pose, success, time.monotonic_ns() - _start_time

    def reset_position(self):
        self.global_pose = np.eye(4, dtype=np.float64)
        cam_rotation_rdf = self.frd_to_rdf_rotation * self.camera_mount_rotation * self.frd_to_rdf_rotation.inv()
        self.global_pose[:3, :3] = cam_rotation_rdf.as_matrix()
        self.previous_transform = np.eye(4, dtype=np.float64)
