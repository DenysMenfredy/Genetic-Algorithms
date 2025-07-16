"""Date utility functions for generating execution names with timestamps."""
from datetime import datetime


def generate_execution_name() -> str:
    """
    Generates a string with the prefix 'execution_' followed by the current date and time
    formatted as YYYYMMDD_HHMMSS.

    Returns:
        A string like 'execution_20250710_011800'
    """
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    execution_name = f"execution_{timestamp}"
    return execution_name
