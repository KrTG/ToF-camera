import os
import time
from typing import Any, Mapping

from pymavlink import mavutil
from serial import Serial

from src import conf

SERIAL_PORT = "/dev/serial0"
BAUD_RATE = 921600
ASYNC_TIMEOUT = 1 / 400 # Should be a bit higher than double the most common message FPS
TIMESYNC_SMOOTHING_ALPHA = 0.05
TIMEOUT = 1 / conf.FPS / 2
LONG_TIMEOUT = 0.5
SYSTEM_ID = 96

def check_connection():
    return os.path.exists(SERIAL_PORT)


def get_connection() -> mavutil.mavserial:
    if not check_connection():
        raise ConnectionError("Serial device not configured.")

    connection = mavutil.mavlink_connection(device=SERIAL_PORT, baud=BAUD_RATE, source_system=SYSTEM_ID)

    assert isinstance(connection, mavutil.mavserial)
    assert isinstance(connection.port, Serial)
    connection.port.xonxoff = False
    connection.port.rtscts  = False
    connection.port.dsrdtr  = False
    if conf.DEBUG:
        print("Loaded MAVLink Version:", connection.WIRE_PROTOCOL_VERSION)
    return connection


class Commander:
    def __init__(self, connection: mavutil.mavfile):
        self.connection = connection

    def wait_heartbeat(self, log=True):
        heartbeat = self.connection.wait_heartbeat()
        if log:
            print(heartbeat)
        return heartbeat

    def send_heartbeat(self):
        if self.connection:
            self.connection.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0,
                0,
                0,
            )

    def set_message_interval(self, message: int, interval_us: int):
        print("<MESSAGE INTERVAL>")
        interval_message = self.connection.mav.command_long_encode(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            message,
            interval_us,
            0,
            0,
            0,
            0,
            0,
        )
        response = None
        while True:
            self.connection.mav.send(interval_message)
            response = self.connection.recv_match(type="COMMAND_ACK", blocking=True, timeout=LONG_TIMEOUT)
            if response is not None:
                break

        print(response)
        print("</MESSAGE INTERVAL>")
        return response.result  # type: ignore

    def odometry(self, x, y, z, qw, qx, qy, qz, timestamp, quality=100, reset_counter=0):
        result = self.connection.mav.odometry_send(
            timestamp,
            mavutil.mavlink.MAV_FRAME_ODOMETRY_NED,
            mavutil.mavlink.MAV_FRAME_BODY_FRD,
            x, y, z,
            qw, qx, qy, qz,
            float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), float("nan"),
            [float("nan")] * 21,
            [float("nan")] * 21,
            reset_counter,
            mavutil.mavlink.MAV_ESTIMATOR_TYPE_VIO,
            quality
        )

        return result


class StateMonitor:
    def __init__(
        self,
        connection: mavutil.mavserial,
        async_messages: list[str],
        sync_messages: list[str],
    ):
        self.connection = connection
        self.async_messages = async_messages
        self.sync_messages = sync_messages
        self.current_state = {}
        self.time_offset_init = False
        self.time_offset = 0

    def is_initialized(self):
        return all(
            msg_type in self.current_state
            for msg_type in (self.async_messages + self.sync_messages)
        )

    def update_state(self) -> Mapping[str, Any]:
        if self.connection:
            self.process_messages(self.sync_messages, self.async_messages)
        return self.current_state

    def reset_buffer(self):
        """
        Reset serial link input buffer - there is too many messages to handle, so just drop them
        to get the latest ones.
        """
        assert isinstance(self.connection.port, Serial)
        self.connection.port.reset_input_buffer()

    def process_messages(self, sync_messages, async_messages):
        if len(sync_messages) == 0 and len(async_messages) == 0:
            return

        if not self.is_initialized():
            sync_left = set(sync_messages) | set(async_messages)
        else:
            sync_left = set(sync_messages)

        async_start = time.monotonic()
        while (time.monotonic() - async_start) < ASYNC_TIMEOUT:
            message = self.connection.recv_msg()
            if message is None:
                break
            msg_type = message.get_type()
            if msg_type in async_messages:
                self.current_state[msg_type] = message
        self.reset_buffer()

        while sync_left:
            message = self.connection.recv_match(blocking=True, timeout=TIMEOUT)
            msg_type = None if message is None else message.get_type()
            if msg_type in sync_left:
                self.current_state[msg_type] = message
                sync_left.remove(msg_type)

    def timesync(self):
        local_time_sent = time.monotonic_ns()
        self.connection.mav.timesync_send(0, local_time_sent)

        msg = self.connection.recv_match(type="TIMESYNC", blocking=True, timeout=TIMEOUT)
        if msg and msg.tc1 and msg.ts1 == local_time_sent:
            local_time_received = time.monotonic_ns()

            rtt = local_time_received - local_time_sent
            delta = msg.tc1 - msg.ts1
            offset = delta - rtt // 2
            if self.time_offset_init:
                a = TIMESYNC_SMOOTHING_ALPHA
                self.time_offset = int((a * offset) + ((1 - a) * self.time_offset))
            else:
                self.time_offset = offset
                self.time_offset_init = True

    def time_ns(self):
        return time.monotonic_ns() + self.time_offset

    def wait_heartbeat(self):
        self.current_state["HEARTBEAT"] = self.connection.recv_match(
            type="HEARTBEAT", blocking=True
        )

    @property
    def status(self):
        out = self.heartbeat.system_status
        return out

    @property
    def landed_state(self):
        out = self.extended_sys_state.landed_state
        return out

    @property
    def manual_mode(self):
        return bool(self.heartbeat.base_mode & 64)

    @property
    def voltage(self):
        out = self.sys_status.voltage_battery
        return out

    @property
    def armed(self):
        return bool(self.heartbeat.base_mode & 128)

    @property
    def landed(self):
        return (
            self.extended_sys_state.landed_state
            == mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND
        )

    @property
    def flying(self):
        return (
            self.extended_sys_state.landed_state
            == mavutil.mavlink.MAV_LANDED_STATE_IN_AIR
        )

    def __getattr__(self, attr) -> Any:
        if attr.upper() in self.current_state:
            return self.current_state[attr.upper()]
        else:
            return None

    def __str__(self):
        return "\n".join(str(msg) for msg in self.current_state.values())


if __name__ == "__main__":
    try:
        connection = get_connection()
        commander = Commander(connection)
        state = StateMonitor(
            connection,
            async_messages=["HEARTBEAT", "SYS_STATUS"],
            sync_messages=["ATTITUDE_QUATERNION"],
        )

        commander.send_heartbeat()
        commander.wait_heartbeat()
        commander.set_message_interval(
            mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_QUATERNION, 6500
        )  # 150 FPS

        _time = time.monotonic()
        i = 0
        while True:
            i += 1
            commander.send_heartbeat()
            state.update_state()

            if i % 100 == 0:
                print(f"FPS: {1 / (time.monotonic() - _time) * 100:.0f}")
                _time = time.monotonic()
                print(f"Voltage: {state.voltage / 1000 / 4}")
                print(
                    f"Attitude quat: q1:{state.attitude_quaternion.q1} q2:{state.attitude_quaternion.q1} q3: {state.attitude_quaternion.q3} q4: {state.attitude_quaternion.q4}"
                )
    except ConnectionError:
        print("Connection lost.")
    finally:
        try:
            connection.close()
        except Exception:
            pass
