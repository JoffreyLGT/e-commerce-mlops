"""Logging configurations."""

import os
import sys
from logging import config

CONSOLE_LOG_LEVEL = os.getenv("CONSOLE_LOG_LEVEL", "WARNING")
TARGET_ENV = os.getenv("TARGET_ENV", "development")

logging_config = {
    "version": 1,
    "formatters": {
        "json": {
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(process)s %(levelname)s %(name)s %(funcName)s %(lineno)s %(message)s",
        },
        "normal": {
            "format": "%(asctime)s %(levelname)s %(name)s, line %(lineno)s in %(funcName)s(): %(message)s"
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
            "filename": "./logs/datascience-log.json",
            "encoding": "utf-8",
            "when": "midnight",
            "interval": 1,
            "utc": True,
        },
    },
    "root": {"level": "DEBUG", "handlers": ["console", "file"], "propagate": True},
}

config.dictConfig(logging_config)
