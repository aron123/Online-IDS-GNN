import logging
import os
import sys
from datetime import datetime

def init_logger(experiment_label=""):
    os.makedirs("logs", exist_ok=True)

    # Create timestamped log filename
    log_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
    if len(experiment_label) > 0:
        log_filename = f"{log_filename}-{experiment_label}"
    log_filename = f"logs/{log_filename}.log"

    # Set up the logger
    logger = logging.getLogger("custom_logger")
    logger.setLevel(logging.DEBUG)

    # Formatter for log messages
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)

    # Stream (stdout) handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # Avoid duplicate logs if init_logger() is called multiple times
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
