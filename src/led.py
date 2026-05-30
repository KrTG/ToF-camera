import gpiozero

class MockLed(gpiozero.LED):
    def __init__(self):
        pass

    def on(self):
        pass

    def off(self):
        pass

    def close(self):
        pass

def get_blue():
    try:
        return gpiozero.LED(23)
    except Exception:
        return MockLed()

def get_green():
    try:
        return gpiozero.LED(24)
    except Exception:
        return MockLed()
