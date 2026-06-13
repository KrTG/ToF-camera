import os
import time

import cv2
from cv2.typing import MatLike
import numpy as np
from cv2.rgbd import OdometryFrame
from scipy.linalg import expm, logm
from scipy.spatial.transform import Rotation

from src import conf
from src.log import get_logger


QUALITY_SMOOTHING_ALPHA = 0.08


logger = get_logger(__name__)

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
        frame_size,
        min_depth=conf.ICPO_MIN_DEPTH,
        max_depth=conf.ICPO_MAX_DEPTH,
        max_depth_diff=conf.ICPO_MAX_DEPTH_DIFF,
        max_points_part=conf.ICPO_MAX_POINTS_PART,
        iter_counts=conf.ICPO_ITER_COUNTS,
        gradient_magnitudes=conf.ICPO_GRADIENT_MAGNITUDES,
        debug_frames_path: str | None = None,
    ):
        self.debug_frames_path = debug_frames_path
        if self.debug_frames_path:
            os.makedirs(self.debug_frames_path, exist_ok=True)

        self.cam_matrix = cam_matrix.astype(np.float32)
        self.cam_matrix_inv = np.linalg.inv(self.cam_matrix).astype(np.float32)
        self.frame_size = frame_size

        self.icpo = cv2.rgbd.RgbdICPOdometry.create(
            cameraMatrix=cam_matrix,
            minDepth=min_depth,
            maxDepth=max_depth,
            maxDepthDiff=max_depth_diff,
            maxPointsPart=max_points_part,
            iterCounts=iter_counts,
            minGradientMagnitudes=gradient_magnitudes,
            #transformType=cv2.rgbd.ODOMETRY_TRANSLATION,
            transformType=cv2.rgbd.ODOMETRY_RIGID_BODY_MOTION
        )

        if conf.DEBUG:
            logger.debug("----ICPO SETTINGS----")
            logger.debug(f"Camera matrix: {self.icpo.getCameraMatrix()}")
            logger.debug(f"Min depth: {self.icpo.getMinDepth()}")
            logger.debug(f"Max depth: {self.icpo.getMaxDepth()}")
            logger.debug(f"Max depth diff: {self.icpo.getMaxDepthDiff()}")
            logger.debug(f"Max points part: {self.icpo.getMaxPointsPart()}")
            logger.debug(f"Iter counts: {self.icpo.getIterationCounts()}")
            logger.debug(f"Min gradient magnitudes: {self.icpo.getMinGradientMagnitudes()}")
            logger.debug(f"Transform type: {self.icpo.getTransformType()}")
            logger.debug("--------")

        self.anchor_attitude: Rotation | None = None
        self.previous_transform: np.ndarray = np.eye(4, dtype=np.float64)

        u_coords = np.arange(self.frame_size[0], dtype=np.float32)
        v_coords = np.arange(self.frame_size[1], dtype=np.float32)
        self.umesh, self.vmesh = np.meshgrid(u_coords, v_coords)

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
        self.cam_frd_rdf_rotation_inv = self.camera_mount_rotation * self.frd_to_rdf_rotation_inv
        self.frd_to_rdf_cam_rotation_inv = self.frd_to_rdf_rotation * self.camera_mount_rotation.inv()

        self.quality = 0.0

        self.reset_position()

    def prepare_warped_frame(
            self, amplitude: MatLike, depth: MatLike, mask: MatLike, frame_id: int, rotation: Rotation
        ):
        _start_time = time.monotonic_ns()

        attitude = self.frd_to_rdf_rotation * rotation * self.cam_frd_rdf_rotation_inv
        warped_frame = None
        if self.anchor_attitude is not None:
            relative_attitude = self.anchor_attitude.inv() * attitude

            R = relative_attitude.as_matrix().astype(np.float32)
            K = self.cam_matrix
            K_inv = self.cam_matrix_inv

            B = R @ K_inv
            H = K @ B
            rotated_z = B[2, 0] * self.umesh + B[2, 1] * self.vmesh + B[2, 2]
            scaled_depth = depth * rotated_z

            warped_amplitude = cv2.warpPerspective(amplitude, H, (amplitude.shape[1], amplitude.shape[0]), flags=cv2.INTER_AREA)
            warped_depth = cv2.warpPerspective(scaled_depth, H, (depth.shape[1], depth.shape[0]), flags=cv2.INTER_NEAREST)
            warped_mask = cv2.warpPerspective(mask, H, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)

            # DEBUG: Save frames for comparison
            if self.debug_frames_path:
                # Original frame
                cv2.imwrite(f"{self.debug_frames_path}/{frame_id:05d}_unwarped_amplitude.png", amplitude)
                cv2.imwrite(f"{self.debug_frames_path}/{frame_id:05d}_unwarped_depth.png", cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255.0/depth.max()), cv2.COLORMAP_RAINBOW))
                cv2.imwrite(f"{self.debug_frames_path}/{frame_id:05d}_unwarped_mask.png", mask)
                # Warped frame
                cv2.imwrite(f"{self.debug_frames_path}/{frame_id:05d}_warped_amplitude.png", warped_amplitude)
                cv2.imwrite(f"{self.debug_frames_path}/{frame_id:05d}_warped_depth.png", cv2.applyColorMap(cv2.convertScaleAbs(warped_depth, alpha=255.0/warped_depth.max()), cv2.COLORMAP_RAINBOW))
                cv2.imwrite(f"{self.debug_frames_path}/{frame_id:05d}_warped_mask.png", warped_mask)

            warped_frame = cv2.rgbd.OdometryFrame.create(
                warped_amplitude, warped_depth, warped_mask, None, frame_id
            )
            self.icpo.prepareFrameCache(warped_frame, cv2.rgbd.ODOMETRY_FRAME_CACHE_DST)
        self.anchor_attitude = attitude
        return warped_frame, time.monotonic_ns() - _start_time

    def integrate_quality_mask(self, mask: np.ndarray):
        """
        Calculate the quality per frame and integrate it into the
        exponential moving average.
        """
        ratio_unmasked = np.count_nonzero(mask) / mask.size

        a = QUALITY_SMOOTHING_ALPHA
        self.quality = (a * ratio_unmasked) + ((1 - a) * self.quality)

        return self.quality

    def integrate_quality_success(self, success: bool):
        """
        If not successful assume 0 quality as a second measure.
        """
        if not success:
            a = QUALITY_SMOOTHING_ALPHA
            self.quality = (1 - a) * self.quality

        return self.quality

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

    def compute_frame(self, anchor_frame: OdometryFrame, warped_frame: OdometryFrame, rotation: Rotation, acceleration: np.ndarray):
        """
        @param anchor_frame: Unwarped previous frame (anchor)
        @param warped_frame: Current warped frame
        """
        skip = warped_frame.ID - anchor_frame.ID
        if skip != 1:
            logger.warning("Computing with skipped frames!")

        _start_time = time.monotonic_ns()

        init_rt = self.previous_transform.copy()
        if skip != 1:
            init_rt = np.linalg.matrix_power(init_rt, skip)
        init_rt[:3, :3] = Rotation.identity().as_matrix()

        dt = skip / conf.FPS
        a_corrected = acceleration + rotation.inv().apply([0, 0, conf.GRAVITY])
        a_rdf = self.frd_to_rdf_cam_rotation_inv.apply(a_corrected)

        correction = 0.5 * a_rdf * dt * dt

        # Subtract the correction - init_rt direction is inverted
        init_rt[:3, 3] -= correction
        init_rt[:3, 3] *= conf.ICPO_INIT_RT_DAMPING

        success, transform = self.icpo.compute2(
            anchor_frame, warped_frame, initRt=init_rt
        )
        if success:
            self.global_pose @= fast_inversion(transform)
            if skip != 1:
                self.previous_transform = fractional_se3_power(transform, 1.0 / skip)
            else:
                self.previous_transform = transform
        else:
            if conf.DEBUG:
                logger.debug(f"Lost tracking on frame {warped_frame.ID}. Re-set using linear prediction.")
            # Apply the 'guess' as the real prediction since we lost tracking
            # and it's the best compromise
            self.global_pose @= fast_inversion(init_rt)

            # Either reset or set to init_rt - we choose to reset
            self.previous_transform = np.eye(4, dtype=np.float64)

        attitude = self.frd_to_rdf_rotation * rotation * self.cam_frd_rdf_rotation_inv
        self.global_pose[:3, :3] = attitude.as_matrix()

        pose = self.rdf_to_frd_transform @ self.global_pose @ self.final_transform
        return pose, success, time.monotonic_ns() - _start_time

    def reset_position(self):
        self.global_pose = np.eye(4, dtype=np.float64)
        cam_rotation_rdf = self.frd_to_rdf_rotation * self.camera_mount_rotation * self.frd_to_rdf_rotation.inv()
        self.global_pose[:3, :3] = cam_rotation_rdf.as_matrix()
        self.previous_transform = np.eye(4, dtype=np.float64)
