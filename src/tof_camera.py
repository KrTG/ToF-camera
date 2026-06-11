import time

import cv2
import numpy as np
import ArducamDepthCamera as ac

from src import conf
from src.log import get_logger

logger = get_logger(__name__)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class TofCamera:
    def __init__(self, range=conf.RANGE, frame_timeout=200, scale=conf.FRAME_SCALE):
        self.cam = None
        self.range = range
        self.frame_timeout = frame_timeout
        self.started = False
        self.scale = scale

        self.clahe = cv2.createCLAHE(
            clipLimit=conf.FRAME_AMPLITUDE_CLAHE_CLIP_LIMIT,
            tileGridSize=(conf.FRAME_AMPLITUDE_CLAHE_GRID_SIZE, conf.FRAME_AMPLITUDE_CLAHE_GRID_SIZE)
        )

    def start(self):
        logger.info("Arducam Depth Camera Streaming.")
        logger.info(f"  SDK version: {ac.__version__}")

        self.cam = ac.ArducamCamera()

        ret = self.cam.open(ac.Connection.CSI, 0)
        if ret != 0:
            logger.error(f"Failed to open camera. Error code: {ret}")
            return

        ret = self.cam.start(ac.FrameType.DEPTH)
        if ret != 0:
            logger.error(f"Failed to start camera. Error code: {ret}")
            self.cam.close()
            return

        self.cam.setControl(ac.Control.RANGE, self.range)
        self.cam.setControl(ac.Control.AUTO_FRAME_RATE, 0)

        info = self.cam.getCameraInfo()
        logger.info("----CAMERA SETTINGS----")
        logger.info(f"Camera resolution: {info.width}x{info.height}")
        logger.info(f"Device type: {info.device_type}")

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

        logger.info(f"Range: {self.range}")
        logger.info(f"Fmt width: {self.fmt_width}")
        logger.info(f"Fmt height: {self.fmt_height}")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Frame mode: {self.frame_mode}")
        logger.info(f"Exposure: {self.exposure}")
        logger.info(f"Frame rate: {self.frame_rate}")
        logger.info(f"Skip frame: {self.skip_frame}")
        logger.info(f"Skip frame loop: {self.skip_frame_loop}")
        logger.info(f"Auto frame rate: {self.auto_frame_rate}")
        logger.info(f"Intrinsic FX: {self.fx}")
        logger.info(f"Intrinsic FY: {self.fy}")
        logger.info(f"Intrinsic CX: {self.cx}")
        logger.info(f"Intrinsic CY: {self.cy}")
        logger.info(f"Denoise: {self.denoise}")
        logger.info("--------")

        self.started = True

    def stop(self):
        if not self.started or not self.cam:
            logger.warning("Camera not initalized.")
            return
        self.cam.stop()
        self.cam.close()

    def get_intrinsic_matrix(self):
        if not self.started or not self.fx or not self.fy or not self.cx or not self.cy:
            logger.warning("Camera not initalized.")
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
            logger.warning("Camera not initalized.")
            return

        frame = self.cam.requestFrame(self.frame_timeout)
        if frame is not None and isinstance(frame, ac.DepthData):
            return frame

    def release_frame_raw(self, frame):
        if hasattr(frame, '__mock__'):
            return
        if not self.started or not self.cam or not self.range:
            logger.warning("Camera not initalized.")
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
        depth = cv2.medianBlur(depth, conf.FRAME_DEPTH_BLUR_WIDTH)
        return depth

    def get_frame_amplitude(self, frame: ac.DepthData):
        """
        Returns normalized amplitude data
        """
        alpha = conf.FRAME_AMPLITUDE_BRIGHTNESS

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
        amplitude = cv2.medianBlur(amplitude, conf.FRAME_AMPLITUDE_BLUR_WIDTH)
        amplitude = self.clahe.apply(amplitude)
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

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def get_frame_rgbd(self, frame: ac.DepthData):
        _start_time = time.monotonic_ns()

        amplitude = self.get_frame_amplitude(frame)
        depth = self.get_frame_depth(frame)
        mask = self.get_frame_mask(frame)

        return amplitude, depth, mask, time.monotonic_ns() - _start_time

    def convert_rgb(self, frame: np.ndarray, mask: np.ndarray):
        result = np.clip(frame * (255.0 / self.range * 1000), 0, 255).astype(
            np.uint8
        )
        result = cv2.applyColorMap(result, cv2.COLORMAP_RAINBOW)
        result[mask == 0] = 0

        return result

    def convert_grayscale(self, frame: np.ndarray, mask: np.ndarray):
        result = frame[:]
        result[mask == 0] = 0
        return frame
