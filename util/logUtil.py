# util/logUtil.py
from loguru import logger
import sys
import os

_LOGGER_CONFIGURED = False
from config import LoggerConfig
cfg = LoggerConfig()

def setup_logger(
    level: str = cfg.level,
    log_file: str | None = cfg.log_file,
):
    """
    Setup loguru logger.
    Should be called ONCE in main process.
    Child processes must NOT call this again.
    """
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return logger

    logger.remove()

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "{process.name}({process.id}) | "
        "<level>{message}</level>"
    )

    # Console sink
    logger.add(
        sys.stderr,
        level=level,
        format=fmt,
        enqueue=True,        # ✅ multiprocessing safe
        backtrace=False,
        diagnose=False,
    )

    # Optional file sink
    if log_file:
        logger.add(
            log_file,
            level=level,
            format=fmt,
            enqueue=True,
            rotation="100 MB",
            retention="7 days",
        )

    _LOGGER_CONFIGURED = True
    return logger


def get_logger():
    """
    Safe to call in child processes.
    Assumes setup_logger() was already called in main.
    """
    return logger
