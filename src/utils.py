import logging
import time
from pathlib import Path

import cv2
import numpy as np

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: A logger instance.
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger


def ensure_dir(dir_path: str | Path) -> Path:
    """
    Ensure that a directory exists. If it doesn't exist, create it.

    Args:
        dir_path (str | Path): The path to the directory.

    Returns:
        Path: The path to the directory.
    """
    directory = Path(dir_path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_image_bgr(image_path: str | Path) -> np.ndarray:
    """
    Read an image from the specified path in BGR format.

    Args:
        image_path (str | Path): The path to the image file.
    
    Returns:
        np.ndarray: The image as a NumPy array in BGR format.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    
    if image is None:
        raise FileNotFoundError(f"Failed to read image")
    
    return image


def save_image(image_path: str | Path, image: np.ndarray):
    """
    Save an image to the specified path.

    Args:
        image_path (str | Path): The path to save the image.
        image (np.ndarray): The image as a NumPy array.
    """
    output_path = Path(image_path)
    ensure_dir(output_path.parent)
    
    success = cv2.imwrite(str(output_path), image)
    
    if not success:
        raise RuntimeError(f"Failed to save image")


def safe_filename(path: str | Path) -> str:
    return Path(path).name


class Timer:
    def __init__(self):
        self.elapsed_ms = 0.0
        self._start_time = 0.0
    
    
    def __enter__(self) -> "Timer":
        self._start_time = time.perf_counter()
        return self
    
    
    def __exit__(self, *_args: object):
        elapsed_seconds = time.perf_counter() - self._start_time
        self.elapsed_ms = elapsed_seconds * 1000.0