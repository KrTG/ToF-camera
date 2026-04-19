import statistics
import time

import ArducamDepthCamera as ac

from src.tof_camera import TofCamera


camera = TofCamera()
camera.start()

last_frame_time = None
frame_times = []
frame_number = 0
while True:
    frame = camera.get_frame_raw(timeout=1200)
    if frame is not None and isinstance(frame, ac.DepthData):
        time_now = time.monotonic_ns()
        if last_frame_time is not None and frame_number > 10:
            frame_time = time_now - last_frame_time
            frame_times.append(frame_time)

        last_frame_time = time_now
        frame_number += 1
        camera.release_frame_raw(frame)
    else:
        raise RuntimeError(f"Frames should not be getting dropped: frame == {frame}")
    if (frame_number + 1) % 100 == 0:
        print(f"Frame: {frame_number}")
        print(f"Mean frame time: {statistics.mean(frame_times) / 1000000} ms")
        print(f"Min frame time: {min(frame_times) / 1000000} ms")
        print(f"Max frame time: {max(frame_times) / 1000000} ms")
        frame_times.clear()
