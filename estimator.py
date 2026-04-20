from src.icpo import IcpOdometry
from src.tof_camera import TofCamera


camera = TofCamera(frame_timeout=1200)
camera.start()

odometry = IcpOdometry(camera.get_intrinsic_matrix())

last_frame_time = None
frame_times = []
frame_number = 0

while True:
    amplitude, depth, mask, prep_time = camera.get_frame_rgbd()

    if amplitude is not None and depth is not None:
        global_pose, odo_time = odometry.next_frame(amplitude, depth, mask, frame_number)

        frame_number += 1
    else:
        raise RuntimeError("Frames should not be getting dropped.")

    if (frame_number + 1) % 100 == 0:
        print("Time budget: 33 ms")
        print(f"Pre-processing time: {prep_time / 1000000} ms")
        print(f"Processing time: {odo_time / 1000000} ms")
