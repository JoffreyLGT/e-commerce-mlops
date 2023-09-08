"""Filter functions used by logging.Logger in logging_config."""


import logging
from collections.abc import Callable
from typing import Literal


def filter_maker(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    order: Literal["Ascending", "Descending"],
) -> Callable:
    """Return a filter object to filter by level.

    Args:
        level: of criticity.
        order: ascending to filer all criticity up or down the provided level.

    Return:
        Filter function.
    """
    level = getattr(logging, level)

    def filter(record) -> bool:
        if order == "Ascending":
            return record.levelno >= level
        return record.levelno <= level

    return filter
