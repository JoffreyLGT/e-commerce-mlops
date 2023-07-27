"""
Import all the models, so that Base has access to them before being imported by Alembic.
Used by alembic to generate automatic migration.
"""

# pylint: disable=W0611
from app.database.base_class import Base
from app.models import User, PredictionFeedback
