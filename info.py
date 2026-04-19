from src import conf
from src.tof_camera import TofCamera

cam = TofCamera(range=conf.RANGE)
cam.start()
cam.stop()
