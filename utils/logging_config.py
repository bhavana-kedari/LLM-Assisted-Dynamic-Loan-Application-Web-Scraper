"""
Configure simple logging for the application.
"""

import logging
import sys


def configure_logging(level: int = logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    ch.setFormatter(formatter)
    logger.handlers = [ch]
