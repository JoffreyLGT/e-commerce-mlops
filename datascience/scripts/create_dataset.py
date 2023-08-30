"""Create a new dataset from remaining dataset.

Created dataset is removed from remaining dataset.
"""
import logging

from src.core.settings import get_dataset_settings
from src.utilities.dataset_utils import get_remaining_dataset_path, load_dataset

settings = get_dataset_settings()
logger = logging.getLogger(__name__)

dataset_path = get_remaining_dataset_path()
features, target = load_dataset(dataset_path)
