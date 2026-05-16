from pymavlink import mavutil

from src import mav
from src.tof_camera import TofCamera
from src.icpo import IcpOdometry
from src.estimator import CameraThread, PrepareCacheThread, ComputeThread, OutputMavlinkThread

def main():
    mav_connection = mav.get_connection()
    heartbeat = mav_connection.wait_heartbeat(timeout=3)
    if heartbeat is None:
        raise RuntimeError("Onboard odometry sensor needs a working mavlink connection.")
    else:
        mav.Commander(mav_connection).set_message_interval(
            mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_QUATERNION, 6500
        )  # 150 FPS

    camera = TofCamera(frame_timeout=0)
    camera.start()
    odometry = IcpOdometry(camera.get_intrinsic_matrix())
    camera_thread = CameraThread(camera, mav_connection=mav_connection)
    prepare_frame_thread = PrepareCacheThread(camera_thread, odometry)
    compute_thread = ComputeThread(prepare_frame_thread, odometry)
    output_thread = OutputMavlinkThread(compute_thread, mav_connection=mav_connection)

    camera_thread.start()
    prepare_frame_thread.start()
    compute_thread.start()
    output_thread.start()

    try:
        while True:
            pass
    finally:
        camera_thread.running = False
        prepare_frame_thread.running = False
        compute_thread.running = False
        output_thread.running = False
        camera_thread.join()
        prepare_frame_thread.join()
        compute_thread.join()
        output_thread.join()


if __name__ == "__main__":
    main()
