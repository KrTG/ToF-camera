import os
from pathlib import Path
import pickle
import numpy as np
from src.icpo import IcpOdometry
from src.tof_camera import TofCamera

def get_test_info(readme_path):
    with open(readme_path, "r") as f:
        content = f.read().strip()
    return content

def get_poses(test_filename, external_rotation=True):
    camera = TofCamera()

    intrinsic_matrix = np.array([
        [190.92, 0, 120.00],
        [0, 19.125, 90.00],
        [0, 0, 1]
    ])
    poses = []
    success_count = 0
    odometry = IcpOdometry(intrinsic_matrix)
    with open(test_filename, "rb") as f:
        frame_idx = 0
        while True:
            try:
                frame_data = pickle.load(f)
                raw_frame, extra_data = frame_data
                amplitude, depth, mask, _ = camera.get_frame_rgbd(raw_frame)
                odometry_frame, _ = odometry.prepare_frame(amplitude, depth, mask, frame_idx)
                if external_rotation:
                    pose, locked, _, _, _ = odometry.compute_frame(odometry_frame, extra_data.get("ROTATION"))
                else:
                    pose, locked, _, _, _ = odometry.compute_frame(odometry_frame)
                poses.append(pose)
                success_count += locked
                frame_idx += 1
            except EOFError:
                break
    return poses, success_count


def test_hover(filename):
    poses, success = get_poses(filename)
    translations = [pose[:3, 3] for pose in poses]

    drifts = [np.linalg.norm(translations[i] - translations[i - 1]) for i in range(1, len(translations))]
    print(f"Hover test {filename} results:")
    print(f"\tframes:\t\t{len(poses)}")
    print(f"\tsucceeded:\t\t{success}")
    print(f"\tmax_step:\t\t{max(drifts):.3f}")
    print(f"\tavg_step:\t\t{sum(drifts) / len(drifts):.3f}")
    print(f"\tstd_step:\t\t{np.std(drifts):.3f}")
    print(f"\tloop_distance:\t\t{np.linalg.norm(translations[-1] - translations[0]):.3f}")
    print(f"\tloop_distance_X:\t\t{np.linalg.norm(translations[-1][0] - translations[0][0]):.3f}")
    print(f"\tloop_distance_Y:\t\t{np.linalg.norm(translations[-1][1] - translations[0][1]):.3f}")
    print(f"\tloop_distance_Z:\t\t{np.linalg.norm(translations[-1][2] - translations[0][2]):.3f}")


def test_translation_z(filename):
    poses, success = get_poses(filename)
    translations = [pose[:3, 3] for pose in poses]

    # Drifts calculated along the XY plane
    drifts = [np.linalg.norm(translations[i][:2] - translations[i - 1][:2]) for i in range(1, len(translations))]
    print(f"Translation Z test {filename} results:")
    print(f"\tframes:\t\t{len(poses)}")
    print(f"\tsucceeded:\t\t{success}")
    print(f"\tmax_step:\t\t{max(drifts):.3f}")
    print(f"\tavg_step:\t\t{sum(drifts) / len(drifts):.3f}")
    print(f"\tstd_step:\t\t{np.std(drifts):.3f}")
    print(f"\tloop_distance_XY:\t\t{np.linalg.norm(translations[-1][:2] - translations[0][:2]):.3f}")
    print(f"\tloop_distance_Z:\t\t{np.linalg.norm(translations[-1][2] - translations[0][2]):.3f}")


def test_translation_xy(filename):
    poses, success = get_poses(filename)
    translations = [pose[:3, 3] for pose in poses]

    # Drifts calculated along the Z axis
    drifts = [np.linalg.norm(translations[i][2] - translations[i - 1][2]) for i in range(1, len(translations))]
    print(f"Translation XY test {filename} results:")
    print(f"\tframes:\t\t{len(poses)}")
    print(f"\tsucceeded:\t\t{success}")
    print(f"\tmax_step:\t\t{max(drifts):.3f}")
    print(f"\tavg_step:\t\t{sum(drifts) / len(drifts):.3f}")
    print(f"\tstd_step:\t\t{np.std(drifts):.3f}")
    print(f"\tloop_distance_XY:\t\t{np.linalg.norm(translations[-1][:2] - translations[0][:2]):.3f}")
    print(f"\tloop_distance_Z:\t\t{np.linalg.norm(translations[-1][2] - translations[0][2]):.3f}")


def test_loop(filename):
    poses, success = get_poses(filename)
    translations = [pose[:3, 3] for pose in poses]

    print(f"Loop test {filename} results:")
    print(f"\tframes:\t\t{len(poses)}")
    print(f"\tsucceeded:\t\t{success}")
    print(f"\tloop_distance:\t\t{np.linalg.norm(translations[-1] - translations[0]):.3f}")

def test_rotation(filename):
    poses, success = get_poses(filename)

    translations = [pose[:3, 3] for pose in poses]
    drifts = [np.linalg.norm(translations[i] - translations[i - 1]) for i in range(1, len(translations))]

    print(f"Rotation test {filename} results:")
    print(f"\tframes:\t\t{len(poses)}")
    print(f"\tsucceeded:\t\t{success}")
    print(f"\tmax_step:\t\t{max(drifts):.3f}")
    print(f"\tavg_step:\t\t{sum(drifts) / len(drifts):.3f}")
    print(f"\tstd_step:\t\t{np.std(drifts):.3f}")
    print(f"\tloop_distance:\t\t{np.linalg.norm(translations[-1] - translations[0]):.3f}")
    print(f"\tloop_distance_X:\t\t{np.linalg.norm(translations[-1][0] - translations[0][0]):.3f}")
    print(f"\tloop_distance_Y:\t\t{np.linalg.norm(translations[-1][1] - translations[0][1]):.3f}")
    print(f"\tloop_distance_Z:\t\t{np.linalg.norm(translations[-1][2] - translations[0][2]):.3f}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename")

    args = parser.parse_args()

    filename = None
    if args.filename:
        filename = args.filename

    for suite in Path("test").iterdir():
        if suite.name == "hover":
            f = test_hover
        elif suite.name == "translation_z":
            f = test_translation_z
        elif suite.name == "translation_xy":
            f = test_translation_xy
        elif suite.name == "loop":
            f = test_loop
        elif suite.name.startswith("rotation"):
            f = test_rotation
        else:
            continue

        for t in suite.glob("*.replay"):
            if filename is not None and t.name != filename and t.name.removesuffix(".replay") != filename:
                continue
            f(t)
