import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def interpolate(msg_0, time_0: float, time_t: float, msg_1, time_1: float):
    assert time_0 < time_t < time_1

    att_0 = np.array([msg_0.q1, msg_0.q2, msg_0.q3, msg_0.q4])
    att_1 = np.array([msg_1.q1, msg_1.q2, msg_1.q3, msg_1.q4])
    rots = Rotation.from_quat([att_0, att_1], scalar_first=True)
    times = [time_0, time_1]
    slerp = Slerp(times, rots)
    interp = slerp(time_t)

    return interp
