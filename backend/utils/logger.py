"""
backend/utils/logger.py
────────────────────────────────────────────────────────────
Structured, application-wide logger.
All modules import from here to get a consistent format.
"""
import logging
import sys
def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with a consistent format.
    Call once per module:  logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)
    # Avoid adding duplicate handlers on repeated imports
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    # Prevent log records from bubbling up to the root logger
    logger.propagate = False
    return logger