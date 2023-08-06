"""Import classes from all files in the crud modules.

Makes it easier to import classes with the syntax. 

Typical usage example:

    from app.crud import user, prediction_feedback
"""

from .crud_prediction_feedback import prediction_feedback
from .crud_user import user
