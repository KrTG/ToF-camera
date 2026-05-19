import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from src import conf
from src.log import get_logger

logger = get_logger(__name__)

def interpolate(msg_0, msg_1, time_t):
    """
    Interpolate attitude, times in miliseconds
    """
    assert msg_0.time_boot_ms < msg_1.time_boot_ms
    att_0 = np.array([msg_0.q1, msg_0.q2, msg_0.q3, msg_0.q4])
    att_1 = np.array([msg_1.q1, msg_1.q2, msg_1.q3, msg_1.q4])
    rots = Rotation.from_quat([att_0, att_1], scalar_first=True)
    if msg_1.time_boot_ms <= time_t:
        if conf.DEBUG:
            logger.debug(f"{msg_0.time_boot_ms} <= {time_t} <= {msg_1.time_boot_ms}")
        return rots[1]
    elif msg_0.time_boot_ms >= time_t:
        if conf.DEBUG:
            logger.debug(f"{msg_0.time_boot_ms} <= {time_t} <= {msg_1.time_boot_ms}")
        return rots[0]
    else:
        times = [msg_0.time_boot_ms, msg_1.time_boot_ms]
        slerp = Slerp(times, rots)
        interp = slerp(time_t)
        return interp
