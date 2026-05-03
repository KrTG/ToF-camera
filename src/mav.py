import os
import time

from pymavlink import mavutil

SERIAL_PORT = "/dev/serial0"
BAUD_RATE = 921600


def check_connection():
    return os.path.exists(SERIAL_PORT)

def get_connection() -> mavutil.mavserial:
    if not check_connection():
        raise ConnectionError("Serial device not configured.")

    connection = mavutil.mavlink_connection(
        device=SERIAL_PORT,
        baud=BAUD_RATE
    )

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
                mavutil.mavlink.MAV_TYPE_ONBOARD,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0,
                0,
                0,
            )


if __name__ == "__main__":
    try:
        connection = get_connection()
        commander = Commander(connection)

        while True:
            print("Sending heartbeat.")
            commander.send_heartbeat()
            commander.wait_heartbeat()
            time.sleep(1)

    except ConnectionError:
        print("Connection lost.")
