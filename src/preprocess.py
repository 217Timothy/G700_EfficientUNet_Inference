from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import (
    INPUT_IMAGE_DIR,
    DEFAULT_MEAN,
    DEFAULT_STD,
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_IMAGE_PIXELS,
)
from src.utils import read_image_bgr


@dataclass(frozen=True)
class LetterboxMeta:
    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    pad_top: int
    pad_left: int


@dataclass(frozen=True)
class PreprocessResult:
    tensor: np.ndarray
    original_bgr: np.ndarray
    letterbox: LetterboxMeta


def validate_image_path(image_path: str | Path) -> Path:
    """
    Validate the image path and ensure it exists and has a valid extension.

    Args:
        image_path (str | Path): The path to the image file.

    Returns:
        Path: The validated image path.
    """
    path = Path(image_path).resolve()
    input_root = Path(INPUT_IMAGE_DIR).resolve()
    
    try:
        path.relative_to(input_root)
    except ValueError as e:
        raise ValueError(f"Input image must be under the input directory.") from e
    
    if not path.is_file():
        raise FileNotFoundError(f"Input image file does not exist")
    
    if path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(f"Input image must have a valid extension")
    
    return path


def validate_image_array(image: np.ndarray):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel BGR image")
    
    height, width = image.shape[:2]
    
    if height < 0 or width < 0:
        raise ValueError("Input image dimensions must be positive")
    
    if height * width > MAX_IMAGE_PIXELS:
        raise ValueError("Input image exceeds maximum pixel count")


def load_image(image_path: str | Path) -> np.ndarray:
    """
    Load an image from the specified path and validate it.

    Args:
        image_path (str | Path): The path to the image file.

    Returns:
        np.ndarray: The loaded image array.
    """
    validated_path = validate_image_path(image_path)
    image = read_image_bgr(validated_path)
    validate_image_array(image)
    
    return image


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to RGB format.

    Args:
        image_bgr (np.ndarray): The input BGR image.

    Returns:
        np.ndarray: The converted RGB image.
    """
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def letterbox_image(
    image: np.ndarray,
    target_size: tuple[int, int],
    pad_value: int = 0,
) -> tuple[np.ndarray, LetterboxMeta]:
    """
    Resize and pad the image to fit the target size while maintaining aspect ratio.

    Args:
        image (np.ndarray): The input image.
        target_size (tuple[int, int]): The target size (height, width).
        pad_value (int): The value to use for padding.

    Returns:
        tuple[np.ndarray, LetterboxMeta]: The letterboxed image and its metadata.
    """
    target_h, target_w = target_size
    original_h, original_w = image.shape[:2]
    
    # Calculate the scaling factor and new size
    scale = min(target_w / original_w, target_h / original_h)
    
    resized_w = int(round(original_w * scale))
    resized_h = int(round(original_h * scale))
    
    
    resized_image = cv2.resize(
        image,
        (resized_w, resized_h),
        interpolation=cv2.INTER_LINEAR,
    )
    
    pad_left = (target_w - resized_w) // 2
    pad_top = (target_h - resized_h) // 2
    
    canvas = np.full(
        (target_h, target_w, image.shape[2]),
        pad_value,
        dtype=image.dtype
    )
    
    canvas[
        pad_top:pad_top + resized_h,
        pad_left:pad_left + resized_w
        :
    ] = resized_image
    
    meta = LetterboxMeta(
        original_size=(original_h, original_w),
        resized_size=(resized_h, resized_w),
        pad_top=pad_top,
        pad_left=pad_left
    )
    
    return canvas, meta


def normalize_image(
    image_rgb: np.ndarray,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
) -> np.ndarray:
    """
    Normalize the image using the specified mean and standard deviation.

    Args:
        image_rgb (np.ndarray): The input RGB image.
        mean (tuple[float, float, float]): The mean values for normalization.
        std (tuple[float, float, float]): The standard deviation values for normalization.

    Returns:
        np.ndarray: The normalized image.
    """
    image = image_rgb.astype(np.float32) / 255.0
    mean_array = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_array = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    
    normalized_image = (image - mean_array) / std_array

    return normalized_image


def to_chw_tensor(image: np.ndarray) -> np.ndarray:
    """
    Convert the image to a CHW tensor format.

    Args:
        image (np.ndarray): The input image in HWC format.

    Returns:
        np.ndarray: The image in CHW format.
    """
    return np.transpose(image, (2, 0, 1))


def add_batch_dimension(tensor: np.ndarray) -> np.ndarray:
    """
    Add a batch dimension to the tensor.

    Args:
        tensor (np.ndarray): The input tensor.

    Returns:
        np.ndarray: The tensor with an added batch dimension.
    """
    return np.expand_dims(tensor, axis=0)


def preprocess_image(
    image_path: str | Path,
    input_size: tuple[int, int],
) -> PreprocessResult:
    """
    Preprocess the image for model inference.

    Args:
        image_path (str | Path): The path to the input image.
        input_size (tuple[int, int]): The target input size for the model.

    Returns:
        PreprocessResult: The preprocessed tensor, original BGR image, and letterbox metadata.
    """
    image_bgr = load_image(image_path)
    image_rgb = bgr_to_rgb(image_bgr)
    letterboxed_image, letterbox_meta = letterbox_image(image_rgb, input_size) 
    normalized_image = normalize_image(letterboxed_image)
    chw_tensor = to_chw_tensor(normalized_image)
    batched_tensor = add_batch_dimension(chw_tensor)
    
    return PreprocessResult(
        tensor=batched_tensor,
        original_bgr=image_bgr,
        letterbox=letterbox_meta,
    )