"""Utility helpers."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure a standard logger for the package."""
    logger = logging.getLogger("instruction_polish")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    return logger
