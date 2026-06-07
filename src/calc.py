import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from src import conf
from src.log import get_logger

logger = get_logger(__name__)

def interpolate_attitude(msg_0, msg_1, time_t_ms):
    """
    Interpolate attitude, times in miliseconds
    """
    assert msg_0.time_boot_ms < msg_1.time_boot_ms
    att_0 = np.array([msg_0.q1, msg_0.q2, msg_0.q3, msg_0.q4])
    att_1 = np.array([msg_1.q1, msg_1.q2, msg_1.q3, msg_1.q4])
    rots = Rotation.from_quat([att_0, att_1], scalar_first=True)
    if msg_1.time_boot_ms <= time_t_ms:
        if conf.DEBUG:
            logger.debug(f"{msg_0.time_boot_ms} <= {time_t_ms} <= {msg_1.time_boot_ms}")
        return rots[1]
    elif msg_0.time_boot_ms >= time_t_ms:
        if conf.DEBUG:
            logger.debug(f"{msg_0.time_boot_ms} <= {time_t_ms} <= {msg_1.time_boot_ms}")
        return rots[0]
    else:
        times = [msg_0.time_boot_ms, msg_1.time_boot_ms]
        slerp = Slerp(times, rots)
        return slerp(time_t_ms)


def interpolate_acceleration(msg_0, msg_1, time_t_usec):
    """
    Interpolate linear values (vx, vy, vz), times in miliseconds
    """
    assert msg_0.time_usec < msg_1.time_usec
    v0 = np.array([msg_0.xacc, msg_0.yacc, msg_0.zacc])
    v1 = np.array([msg_1.xacc, msg_1.yacc, msg_1.zacc])
    if msg_1.time_usec <= time_t_usec:
        if conf.DEBUG:
            logger.debug(f"{msg_0.time_usec} <= {time_t_usec} <= {msg_1.time_usec}")
        return v1
    elif msg_0.time_usec >= time_t_usec:
        if conf.DEBUG:
            logger.debug(f"{msg_0.time_usec} <= {time_t_usec} <= {msg_1.time_usec}")
        return v0
    else:
        alpha = (time_t_usec - msg_0.time_usec) / (msg_1.time_usec - msg_0.time_usec)
        assert 0 < alpha < 1
        return v0 + alpha * (v1 - v0)
