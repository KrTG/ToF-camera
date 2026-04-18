import time

import cv2
import numpy as np
import ArducamDepthCamera as ac

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class UserRect:
    def __init__(self) -> None:
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0

    @property
    def rect(self):
        return (
            self.start_x,
            self.start_y,
            self.end_x - self.start_x,
            self.end_y - self.start_y,
        )

    @property
    def slice(self):
        return (slice(self.start_y, self.end_y), slice(self.start_x, self.end_x))

    @property
    def empty(self):
        return self.start_x == self.end_x and self.start_y == self.end_y


confidence_value = 30
selectRect, followRect = UserRect(), UserRect()


def getPreviewRGB(preview: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    preview = np.nan_to_num(preview)
    preview[confidence < confidence_value] = (0, 0, 0)
    return preview


def on_confidence_changed(value):
    global confidence_value
    confidence_value = value


class TofCamera:
    def __init__(self, conf=None, max_distance=4000):
        self.cam = None
        self.conf = conf
        self.max_distance = max_distance
        self.started = False

    def start(self):
        print("Arducam Depth Camera Streaming.")
        print("  SDK version:", ac.__version__)

        self.cam = ac.ArducamCamera()

        ret = 0
        if self.conf is not None:
            ret = self.cam.openWithFile(self.conf, 0)
        else:
            ret = self.cam.open(ac.Connection.CSI, 0)
        if ret != 0:
            print("Failed to open camera. Error code:", ret)
            return

        ret = self.cam.start(ac.FrameType.DEPTH)
        if ret != 0:
            print("Failed to start camera. Error code:", ret)
            self.cam.close()
            return

        self.cam.setControl(ac.Control.RANGE, self.max_distance)

        info = self.cam.getCameraInfo()
        print(f"Camera resolution: {info.width}x{info.height}")
        print(f"Device type: {info.device_type}")

        self.range = self.cam.getControl(ac.Control.RANGE)
        self.exposure = self.cam.getControl(ac.Control.EXPOSURE)
        self.denoise = self.cam.getControl(ac.Control.DENOISE)
        self.auto_frame_rate = self.cam.getControl(ac.Control.AUTO_FRAME_RATE)
        self.frame_rate = self.cam.getControl(ac.Control.FRAME_RATE)

        print(f"Range: {self.range}")
        print(f"Exposure: {self.exposure}")
        print(f"Denoise: {self.denoise}")
        print(f"Auto frame rate: {self.auto_frame_rate}")
        print(f"Frame rate: {self.frame_rate}")

        self.started = True

    def get_frame(self):
        if not self.started or not self.cam or not self.range:
            print("Camera not initalized.")
            return

        frame = self.cam.requestFrame(200)
        if frame is not None and isinstance(frame, ac.DepthData):
            depth_buf = frame.depth_data
            confidence_buf = frame.confidence_data

            result_image = (depth_buf * (255.0 / self.range)).astype(np.uint8)
            result_image = cv2.applyColorMap(result_image, cv2.COLORMAP_RAINBOW)
            result_image = getPreviewRGB(result_image, confidence_buf)

            cv2.rectangle(result_image, followRect.rect, WHITE, 1)
            if not selectRect.empty:
                cv2.rectangle(result_image, selectRect.rect, BLACK, 2)

            self.cam.releaseFrame(frame)
            return result_image

cam = TofCamera()
frame = None
def stream_frames():
    print("Streaming video")
    global cam
    global frame

    if not cam.started:
        cam.start()

    while True:
        im = cam.get_frame()
        if im is not None:
            imgencode=cv2.imencode('.jpg',im)[1]
            stringData=imgencode.tobytes()
            output = b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n'
            frame = output
        if frame is not None:
            yield frame
            time.sleep(0.033)
