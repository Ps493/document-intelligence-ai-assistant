"""
logger.py
---------
Simple, reusable logging setup.

Why this matters (interview point):
Real production systems don't use print() statements — they use logging
so that events are timestamped, leveled (INFO/ERROR/etc.), and saved to
a file for later debugging. This is a small but genuine "production-ready"
habit that's cheap to add and signals engineering maturity.
"""

import logging
import os
from core.config import LOG_FILE_PATH

# Make sure the logs/ folder exists before we try to write into it.
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger that writes to BOTH the console and a file.

    Usage in any module:
        from core.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if this function is called multiple times
    # (e.g. Streamlit re-runs scripts on every interaction).
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler - so we see logs live while running the app
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler - so logs persist and can be reviewed later
    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
