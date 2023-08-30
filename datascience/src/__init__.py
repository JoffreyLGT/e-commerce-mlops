"""Define src as a module and ensure some files are called automatically.

logging_config will always be imported when running or calling the module.
"""

# To ensure the Logger is configured
import src.core.logging_config
