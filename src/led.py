import gpiozero

def get_blue():
    return gpiozero.LED(23)

def get_green():
    return gpiozero.LED(24)
