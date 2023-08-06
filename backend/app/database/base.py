"""Import all the models.

Ensure Base has access to them before being imported by Alembic.
Used by alembic to generate automatic migration.
"""

from app.database.base_class import Base
from app.models import PredictionFeedback, User
