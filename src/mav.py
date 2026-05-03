import os
import time
from typing import Any, Mapping

from pymavlink import mavutil

SERIAL_PORT = "/dev/serial0"
BAUD_RATE = 921600
TIMEOUT = 3


def check_connection():
    return os.path.exists(SERIAL_PORT)


def get_connection() -> mavutil.mavserial:
    if not check_connection():
        raise ConnectionError("Serial device not configured.")

    connection = mavutil.mavlink_connection(device=SERIAL_PORT, baud=BAUD_RATE)

    assert isinstance(connection, mavutil.mavserial)
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

    def is_initialized(self):
        return all(
            msg_type in self.current_state
            for msg_type in (self.async_messages + self.sync_messages)
        )

    def update_state(self) -> Mapping[str, Any]:
        if self.connection:
            self.process_messages(self.async_messages, sync=False)
            self.process_messages(self.sync_messages, sync=True)
        return self.current_state

    def process_messages(self, messages, sync):
        if len(messages) == 0:
            return

        if not self.is_initialized():
            sync = True

        msg_left = set(messages)
        if sync:
            stime = time.time()
            while msg_left and (time.time() - stime) < TIMEOUT:
                message = self.connection.recv_match(blocking=True, timeout=TIMEOUT)
                msg_type = None if message is None else message.get_type()
                if msg_type in msg_left:
                    self.current_state[msg_type] = message
                    msg_left.remove(msg_type)
        else:
            while msg_left:
                message = self.connection.recv_match(blocking=False, timeout=None)
                if message is None:
                    break
                msg_type = message.get_type()
                if msg_type in messages:
                    self.current_state[msg_type] = message
                    msg_left.discard(msg_type)

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
            connection, async_messages=["HEARTBEAT", "SYS_STATUS"], sync_messages=[]
        )

        while True:
            print("Sending heartbeat.")
            commander.send_heartbeat()
            commander.wait_heartbeat()
            state.update_state()
            print(f"Voltage: {state.voltage / 1000 / 4}")
            time.sleep(1)

    except ConnectionError:
        print("Connection lost.")
