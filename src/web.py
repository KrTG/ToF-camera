import threading
import time

import cv2

from estimator import CameraThread, PipelineThread
from src.tof_camera import TofCamera

frame_lock = threading.Lock()
global_amplitude = None
global_depth = None


class FrameSaverThread(PipelineThread):
    def __init__(
        self, camera_thread: PipelineThread, camera: TofCamera, passthrough=False
    ):
        super().__init__()
        self.camera_thread = camera_thread
        self.camera = camera
        self.passthrough = passthrough

    def run(self):
        global global_amplitude
        global global_depth

        self.running = True
        try:
            while self.running:
                frame = self.camera_thread.get_frame()
                if frame is None:
                    self.running = False
                    break

                # Save the frame globally
                amplitude, depth, mask, _ = frame
                amplitude = self.camera.convert_grayscale(amplitude, mask)
                depth = self.camera.convert_rgb(depth, mask)

                with frame_lock:
                    global_amplitude = amplitude
                    global_depth = depth

                if self.passthrough:
                    with self.condition:
                        if self.frame is not None:
                            print(
                                "Compute: Next frame ready, while previous was not acquired!"
                            )
                            self.running = False
                            break
                        self.frame = frame
                        self.condition.notify()
        finally:
            if self.passthrough:
                with self.condition:
                    self.running = False
                    self.condition.notify_all()


cam_thread = None
frame_saver_thread = None
camera_lock = threading.Lock()


def stream_frames(image="amplitude"):
    global frame_lock
    global global_amplitude
    global global_depth

    global camera_lock
    global cam_thread
    global frame_saver_thread

    print("Streaming video")

    with camera_lock:
        if cam_thread is None:
            camera = TofCamera(frame_timeout=0)
            cam_thread = CameraThread(camera)
            frame_saver_thread = FrameSaverThread(cam_thread, camera)
            camera.start()
            cam_thread.start()
            frame_saver_thread.start()

    while True:
        with frame_lock:
            if image == "amplitude":
                im = global_amplitude
            elif image == "depth":
                im = global_depth
        if im is not None:
            imgencode = cv2.imencode(".jpg", im)[1]
            stringData = imgencode.tobytes()
            output = (
                b"--frame\r\n"
                b"Content-Type: text/plain\r\n\r\n" + stringData + b"\r\n"
            )
            yield output
            time.sleep(0.01)
