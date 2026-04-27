import threading
import time
from unittest import result

import cv2
import numpy as np
import ArducamDepthCamera as ac

from src import conf

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class TofCamera:
    def __init__(self, range=conf.RANGE, frame_timeout=200, scale=1):
        self.cam = None
        self.range = range
        self.frame_timeout = frame_timeout
        self.started = False
        self.scale = scale

        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

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
        self.cam.setControl(ac.Control.FMT_WIDTH, 120)
        self.cam.setControl(ac.Control.FMT_HEIGHT, 90)
        self.cam.setControl(ac.Control.AUTO_FRAME_RATE, 0)

        info = self.cam.getCameraInfo()
        print("----CAMERA SETTINGS----")
        print(f"Camera resolution: {info.width}x{info.height}")
        print(f"Device type: {info.device_type}")

        self.range = self.cam.getControl(ac.Control.RANGE)
        self.fmt_height = self.cam.getControl(ac.Control.FMT_HEIGHT)
        self.fmt_width = self.cam.getControl(ac.Control.FMT_WIDTH)
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

        s = self.scale
        return (
            np.array(
                [
                    [self.fx * s, 0, self.cx * s],
                    [0, self.fy * s, self.cy * s],
                    [0, 0, 100],
                ],
                dtype=np.float32,
            )
            / 100
        )

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

    def get_frame_depth(self, frame: ac.DepthData):
        """
        Returns depth data in meters as float32
        """
        depth = frame.depth_data
        if self.scale != 1:
            depth = cv2.resize(
                src=depth,
                dsize=None,
                dst=None,
                fx=self.scale,
                fy=self.scale,
                interpolation=cv2.INTER_NEAREST_EXACT,
            )
        depth = depth.astype(np.float32) / 1000.0
        return depth

    def get_frame_amplitude(self, frame: ac.DepthData):
        """
        Returns normalized amplitude data
        """
        alpha = 0.12  # TODO: Implement adaptive alpha to adjust for lightning conditions

        amplitude = frame.amplitude_data
        if self.scale != 1:
            amplitude = cv2.resize(
                src=amplitude,
                dsize=None,
                dst=None,
                fx=self.scale,
                fy=self.scale,
                interpolation=cv2.INTER_AREA,
        )
        amplitude = cv2.convertScaleAbs(amplitude, alpha=alpha)
        #amplitude = self.clahe.apply(amplitude)
        return amplitude

    def get_frame_mask(self, frame: ac.DepthData):
        """
        Returns a mask based on a confidence level
        """
        confidence = frame.confidence_data
        if self.scale != 1:
            confidence = cv2.resize(
                src=confidence,
                dsize=None,
                dst=None,
                fx=self.scale,
                fy=self.scale,
                interpolation=cv2.INTER_NEAREST_EXACT,
            )
        mask = (confidence >= conf.ICPO_CONFIDENCE).astype(np.uint8) * 255
        return mask

    def get_frame_rgbd(self, frame: ac.DepthData):
        _start_time = time.monotonic_ns()

        amplitude = self.get_frame_amplitude(frame)
        depth = self.get_frame_depth(frame)
        mask = self.get_frame_mask(frame)

        return amplitude, depth, mask, time.monotonic_ns() - _start_time

    def get_frame_depth_rgb(self, frame: ac.DepthData):
        result = self.get_frame_depth(frame)

        result = np.clip(result * (255.0 / self.range * 1000), 0, 255).astype(
            np.uint8
        )
        result = cv2.applyColorMap(result, cv2.COLORMAP_RAINBOW)
        mask = self.get_frame_mask(frame)
        result[mask == 0] = 0

        return result

    def get_frame_amplitude_grayscale(self, frame: ac.DepthData):
        result = self.get_frame_amplitude(frame)
        mask = self.get_frame_mask(frame)
        result[mask == 0] = 0
        return result
