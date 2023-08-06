"""Seed DB with basic Data to start using the API."""

import logging

from app.database.init_db import init_db
from app.database.session import SessionLocal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init() -> None:
    """Create a DB session and call init_db to seed it."""
    db = SessionLocal()
    init_db(db)


def main() -> None:
    """Main function called when the script starts."""
    logger.info("Creating initial data")
    init()
    logger.info("Initial data created")


if __name__ == "__main__":
    main()
