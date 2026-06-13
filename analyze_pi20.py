import pickle
import numpy as np
from src.icpo import IcpOdometry
from src.tof_camera import TofCamera
from src import conf
from scipy.spatial.transform import Rotation

camera = TofCamera()
intrinsic_matrix = np.array([
    [190.92 * conf.FRAME_SCALE, 0, 120.00 * conf.FRAME_SCALE],
    [0, 190.125 * conf.FRAME_SCALE, 90.00 * conf.FRAME_SCALE],
    [0, 0, 1]
])
odometry = IcpOdometry(intrinsic_matrix, conf.FRAME_SIZE)
anchor_frame = None

print(f"{'Frame':<6} | {'Pitch (deg)':<12} | {'ICP Tx':<8} | {'ICP Ty':<8} | {'ICP Tz':<8} | {'Min Depth':<10} | {'Max Depth':<10} | {'Valid Pts':<10}")
print("-" * 100)

with open("test/rotation_pitch/pi20.replay", "rb") as f:
    frame_idx = 0
    while True:
        try:
            frame_data = pickle.load(f)
            raw_frame, extra_data = frame_data
            rotation = extra_data.get("rotation")
            acceleration = extra_data.get("acceleration")

            euler = rotation.as_euler('xyz', degrees=True)
            roll, pitch, yaw = euler

            amplitude, depth, mask, _ = camera.get_frame_rgbd(raw_frame)

            # depth is in meters
            valid_depths = depth[(depth >= conf.ICPO_MIN_DEPTH) & (depth <= conf.ICPO_MAX_DEPTH)]
            min_d = np.min(valid_depths) if len(valid_depths) > 0 else 0.0
            max_d = np.max(valid_depths) if len(valid_depths) > 0 else 0.0
            valid_count = len(valid_depths)

            warped_frame, _ = odometry.prepare_warped_frame(amplitude, depth, mask, frame_idx, rotation)

            tx, ty, tz = 0.0, 0.0, 0.0
            if anchor_frame is not None and warped_frame is not None:
                pose, success, _, = odometry.compute_frame(anchor_frame, warped_frame, rotation, acceleration)
                transform = odometry.previous_transform
                tx, ty, tz = transform[:3, 3]

                print(f"{frame_idx:<6} | {pitch:<12.2f} | {tx:<8.4f} | {ty:<8.4f} | {tz:<8.4f} | {min_d:<10.3f} | {max_d:<10.3f} | {valid_count:<10}")

            anchor_frame, _ = odometry.prepare_regular_frame(amplitude, depth, mask, frame_idx)
            frame_idx += 1
        except EOFError:
            break
