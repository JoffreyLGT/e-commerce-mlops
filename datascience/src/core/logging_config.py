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
    "disable_existing_loggers": False,  # Update all previous loggers settings
    "formatters": {
        "json": {
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(process)s %(levelname)s %(name)s %(funcName)s %(lineno)s %(filename)s %(message)s",  # noqa: E501
        },
        "normal": {
            "format": "%(levelname)s in %(module)s.%(funcName)s, line %(lineno)s: %(message)s"  # noqa: E501
        },
        "brief": {"format": "%(message)s"},
    },
    "filters": {
        "warning_and_above": {
            "()": "src.core.logging_filters.filter_maker",
            "level": "WARNING",
            "order": "Ascending",
        },
        "info_and_bellow": {
            "()": "src.core.logging_filters.filter_maker",
            "level": "INFO",
            "order": "Descending",
        },
    },
    "handlers": {
        "console": {
            "level": CONSOLE_LOG_LEVEL.upper(),
            "class": "logging.StreamHandler",
            "formatter": "brief",
            "stream": sys.stderr,
            "filters": ["info_and_bellow"],
        },
        "console_detailed": {
            "level": CONSOLE_LOG_LEVEL.upper(),
            "class": "logging.StreamHandler",
            "formatter": "normal",
            "stream": sys.stderr,
            "filters": ["warning_and_above"],
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
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "console_detailed", "file"],
        "propagate": True,
    },
}

config.dictConfig(logging_config)
