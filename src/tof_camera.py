import time

import cv2
import numpy as np
import ArducamDepthCamera as ac

from src import conf

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class TofCamera:
    def __init__(self, range=conf.RANGE, frame_timeout=200):
        self.cam = None
        self.range = range
        self.frame_timeout = frame_timeout
        self.started = False

        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    def start(self):
        print("Arducam Depth Camera Streaming.")
        print("  SDK version:", ac.__version__)

        self.cam = ac.ArducamCamera()

        ret = self.cam.open(ac.Connection.CSI, 0)
        if ret != 0:
            print("Failed to open camera. Error code:", ret)
            return

        ret = self.cam.start(ac.FrameType.DEPTH)
        if ret != 0:
            print("Failed to start camera. Error code:", ret)
            self.cam.close()
            return

        self.cam.setControl(ac.Control.RANGE, self.range)
        self.cam.setControl(ac.Control.AUTO_FRAME_RATE, 0)

        info = self.cam.getCameraInfo()
        print("----CAMERA SETTINGS----")
        print(f"Camera resolution: {info.width}x{info.height}")
        print(f"Device type: {info.device_type}")

        self.range = self.cam.getControl(ac.Control.RANGE)
        self.fmt_height = self.cam.getControl(ac.Control.FMT_WIDTH)
        self.fmt_width = self.cam.getControl(ac.Control.FMT_HEIGHT)
        self.mode = self.cam.getControl(ac.Control.MODE)
        self.frame_mode = self.cam.getControl(ac.Control.FRAME_MODE)
        self.exposure = self.cam.getControl(ac.Control.EXPOSURE)
        self.frame_rate = self.cam.getControl(ac.Control.FRAME_RATE)
        self.skip_frame = self.cam.getControl(ac.Control.SKIP_FRAME)
        self.skip_frame_loop = self.cam.getControl(ac.Control.SKIP_FRAME_LOOP)
        self.auto_frame_rate = self.cam.getControl(ac.Control.AUTO_FRAME_RATE)
        self.fx = self.cam.getControl(ac.Control.INTRINSIC_FX)
        self.fy = self.cam.getControl(ac.Control.INTRINSIC_FY)
        self.cx = self.cam.getControl(ac.Control.INTRINSIC_CX)
        self.cy = self.cam.getControl(ac.Control.INTRINSIC_CY)
        self.denoise = self.cam.getControl(ac.Control.DENOISE)

        print(f"Range: {self.range}")
        print(f"Fmt width: {self.fmt_width}")
        print(f"Fmt height: {self.fmt_height}")
        print(f"Mode: {self.mode}")
        print(f"Frame mode: {self.frame_mode}")
        print(f"Exposure: {self.exposure}")
        print(f"Frame rate: {self.frame_rate}")
        print(f"Skip frame: {self.skip_frame}")
        print(f"Skip frame loop: {self.skip_frame_loop}")
        print(f"Auto frame rate: {self.auto_frame_rate}")
        print(f"Intrinsic FX: {self.fx}")
        print(f"Intrinsic FY: {self.fy}")
        print(f"Intrinsic CX: {self.cx}")
        print(f"Intrinsic CY: {self.cy}")
        print(f"Denoise: {self.denoise}")
        print("--------")

        self.started = True

    def stop(self):
        if not self.started or not self.cam:
            print("Camera not initalized.")
            return
        self.cam.stop()
        self.cam.close()

    def get_intrinsic_matrix(self):
        if not self.started or not self.fx or not self.fy or not self.cx or not self.cy:
            print("Camera not initalized.")
            return

        return np.array([
            [self.fx,   0,          self.cx],
            [0,         self.fy,    self.cy],
            [0,         0,          1]
        ], dtype=np.float32) / 100

    def get_frame_raw(self):
        if not self.started or not self.cam or not self.range:
            print("Camera not initalized.")
            return

        frame = self.cam.requestFrame(self.frame_timeout)
        if frame is not None and isinstance(frame, ac.DepthData):
            return frame

    def release_frame_raw(self, frame):
        if not self.started or not self.cam or not self.range:
            print("Camera not initalized.")
            return
        self.cam.releaseFrame(frame)

    def get_depth_rgb(self):
        if not self.started or not self.cam or not self.range:
            print("Camera not initalized.")
            return

        frame = self.get_frame_raw()
        if frame is not None:
            data = frame.depth_data

            result_image = np.clip(data * (255.0 / self.range), 0, 255).astype(np.uint8)
            result_image = cv2.applyColorMap(result_image, cv2.COLORMAP_RAINBOW)

            self.cam.releaseFrame(frame)
            return result_image

    def get_amplitude_grayscale(self):
        if not self.started or not self.cam or not self.range:
            print("Camera not initalized.")
            return

        frame = self.get_frame_raw()
        if frame is not None:
            data = frame.amplitude_data
            alpha = 0.08 # TODO: Implement adaptive alpha to adjust for lightning conditions
            temp_8u = cv2.convertScaleAbs(data, alpha=alpha)
            result_image = self.clahe.apply(temp_8u)
            self.cam.releaseFrame(frame)
            return result_image

    def get_frame_rgbd(self):
        if not self.started or not self.cam or not self.range:
            print("Camera not initalized.")
            return (None, None)

        frame = self.get_frame_raw()
        if frame is not None:
            alpha = 0.08 # TODO: Implement adaptive alpha to adjust for lightning conditions

            confidence = frame.confidence_data

            amplitude = frame.amplitude_data
            amplitude = cv2.convertScaleAbs(amplitude, alpha=alpha)
            amplitude = self.clahe.apply(amplitude)

            depth = frame.depth_data
            depth = np.where(confidence < 30, 0.0, depth).astype(np.float32) / 1000.0

            self.cam.releaseFrame(frame)

            return amplitude, depth

        return (None, None)


cam = TofCamera()
frame = None


def stream_frames():
    print("Streaming video")
    global cam
    global frame

    if not cam.started:
        cam.start()

    while True:
        im = cam.get_amplitude_grayscale()
        if im is not None:
            imgencode = cv2.imencode(".jpg", im)[1]
            stringData = imgencode.tobytes()
            output = (
                b"--frame\r\n"
                b"Content-Type: text/plain\r\n\r\n" + stringData + b"\r\n"
            )
            frame = output
        if frame is not None:
            yield frame
            time.sleep(0.033)
