from pathlib import Path


def check_path_exists(path_str: str) -> bool:
    """
    Checks if a given file or folder path exists.

    Args:
        path_str: The string representation of the file or folder path.

    Returns:
        True if the path exists, False otherwise.
    """
    path_obj = Path(path_str)
    return path_obj.exists()

def create_folder(path_str: str) -> bool:
    """
    Creates a folder at the specified path. If the folder already exists, does nothing.

    Args:
        path_str: The path of the folder to create.

    Returns:
        True if the folder was created or already exists, False if creation failed.
    """
    try:
        path_obj = Path(path_str)
        path_obj.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating folder '{path_str}': {e}")
        return False
