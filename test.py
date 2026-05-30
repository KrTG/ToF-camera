import os
from pathlib import Path
import pickle
import sys
import numpy as np
from src import conf
from src.icpo import IcpOdometry
from src.tof_camera import TofCamera
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("WebAgg")

def get_test_info(readme_path):
    with open(readme_path, "r") as f:
        content = f.read().strip()
    return content

def get_poses(test_filename):
    camera = TofCamera()

    intrinsic_matrix = np.array([
        [190.92 * conf.FRAME_SCALE, 0, 120.00 * conf.FRAME_SCALE],
        [0, 190.125 * conf.FRAME_SCALE, 90.00 * conf.FRAME_SCALE],
        [0, 0, 1]
    ])
    poses = []
    success_count = 0
    odometry = IcpOdometry(intrinsic_matrix)
    anchor_frame = None
    with open(test_filename, "rb") as f:
        frame_idx = 0
        while True:
            try:
                frame_data = pickle.load(f)
                raw_frame, extra_data = frame_data
                rotation = extra_data.get("rotation") or extra_data.get("ROTATION")

                amplitude, depth, mask, _ = camera.get_frame_rgbd(raw_frame)
                warped_frame, _ = odometry.prepare_warped_frame(amplitude, depth, mask, frame_idx, rotation)
                if anchor_frame is not None and warped_frame is not None:
                    pose, locked, _, = odometry.compute_frame(anchor_frame, warped_frame, rotation)
                    #print(locked)
                    poses.append(pose)
                    success_count += locked
                anchor_frame, _ = odometry.prepare_regular_frame(amplitude, depth, mask, frame_idx)
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
    #for t in translations:
    #    print(t)

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


def plot(filename):
    poses, success = get_poses(filename)

    translations = np.array([pose[:3, 3] for pose in poses])
    xs = translations[:, 0]
    ys = translations[:, 1]
    zs = translations[:, 2]

    # Create a time array for coloring and sizing
    time_values = np.arange(len(poses))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Color points based on time
    colors = matplotlib.colormaps.get_cmap('viridis')(time_values / float(len(poses)))

    # Size points based on time (e.g., from 20 to 200)
    s_values = np.linspace(10, 16, len(poses))

    ax.scatter(xs, ys, zs, c=colors, s=s_values, marker='o') # type: ignore

    # Calculate ranges for consistent scaling
    max_range = np.array([xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()]).max() / 2.0

    mid_x = (xs.max() + xs.min()) * 0.5
    mid_y = (ys.max() + ys.min()) * 0.5
    mid_z = (zs.max() + zs.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Pose Plot for {filename}')
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename")
    parser.add_argument("-p", "--plot", action="store_true")

    args = parser.parse_args()

    filename = None
    if args.filename:
        filename = args.filename

    if args.plot and not args.filename:
        print("Can only plot one file at a time.")
        sys.exit(1)

    for suite in Path("test").iterdir():
        if args.plot:
            f = plot
        elif suite.name == "hover":
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
