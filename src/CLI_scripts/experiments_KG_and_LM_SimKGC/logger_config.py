import logging
import sys
from datetime import datetime

def _setup_logger():
    format_long = '%(asctime)s - %(levelname)s - %(filename)s/%(funcName)s: %(message)s'

    logger_start_time = datetime.now().strftime("%d.%m.%Y_%H:%M")

    logging.basicConfig(format = format_long, level = logging.DEBUG,
                        datefmt = "%d.%m.%Y %H:%M:%S",
                        handlers = [
                            logging.FileHandler(f'log_SimKGC_run_{logger_start_time}.txt', mode = 'w'),
                            logging.StreamHandler(sys.stdout)
                        ])

    logger = logging.getLogger()

    return logger


logger = _setup_logger()
