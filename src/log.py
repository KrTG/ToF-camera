import logging

def get_logger(name="tof-camera"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG) # Set to DEBUG to capture all levels

    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Add formatter to ch
    ch.setFormatter(formatter)

    # Add ch to logger
    if not logger.handlers: # Avoid adding multiple handlers if get_logger is called multiple times
        logger.addHandler(ch)

    return logger
