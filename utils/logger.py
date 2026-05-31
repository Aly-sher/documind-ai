"""
utils/logger.py
---------------
Centralised logging configuration for DocuMind AI.
Uses loguru for structured, levelled logging across all modules.
"""

import sys
from loguru import logger


def setup_logger(log_level: str = "INFO") -> None:
    """
    Configures loguru with a clean format for console output.
    Call once at application startup (in app.py).
    """
    logger.remove()  # Remove default handler

    # Console handler — colourised, human-readable
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True,
    )

    # File handler — rotating log file for debugging
    logger.add(
        "documind.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
        rotation="5 MB",
        retention="7 days",
        compression="zip",
    )


# Re-export so all modules can do: from utils.logger import logger
__all__ = ["logger", "setup_logger"]