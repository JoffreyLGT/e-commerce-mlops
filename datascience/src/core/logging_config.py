"""Logging configurations."""

import sys
from logging import config
from pathlib import Path

from src.core.settings import get_common_settings

settings = get_common_settings()

CONSOLE_LOG_LEVEL = settings.CONSOLE_LOG_LEVEL
TARGET_ENV = settings.TARGET_ENV

logging_config = {
    "version": 1,
    "formatters": {
        "json": {
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(process)s %(levelname)s %(name)s %(funcName)s %(lineno)s %(filename)s %(message)s",  # noqa: E501
        },
        "normal": {
            "format": "%(asctime)s %(levelname)s %(pathname)s, line %(lineno)s in %(funcName)s():\n%(message)s\n"  # noqa: E501
        },
    },
    "handlers": {
        "console": {
            "level": CONSOLE_LOG_LEVEL.upper(),
            "class": "logging.StreamHandler",
            "formatter": "normal",
            "stream": sys.stderr,
        },
        "file": {
            "level": f'{"DEBUG" if TARGET_ENV == "development" else "INFO"}',
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "json",
            "filename": str(Path(settings.LOGS_DIR) / settings.LOGS_FILE_NAME),
            "encoding": "utf-8",
            "when": "midnight",
            "interval": 1,
            "utc": True,
        },
    },
    "root": {"level": "DEBUG", "handlers": ["console", "file"], "propagate": True},
}

config.dictConfig(logging_config)
