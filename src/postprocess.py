from pathlib import Path

import cv2
import numpy as np

from config import (
    CLASS_COLORS_BGR,
    DEFAULT_OVERLAY_ALPHA,
    DEFAULT_THRESHOLD,
)

from src.utils import save_image
from src.preprocess import LetterboxMeta


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """
    Apply the sigmoid function to the input logits.

    Args:
        logits (np.ndarray): The input logits.

    Returns:
        np.ndarray: The output probabilities after applying the sigmoid function.
    """
    return 1 / (1 + np.exp(-logits))


def softmax(logits: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Apply the softmax function to the input logits along the specified axis.

    Args:
        logits (np.ndarray): The input logits.
        axis (int): The axis along which to apply the softmax function.

    Returns:
        np.ndarray: The output probabilities after applying the softmax function.
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_values= np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def squeeze_prediction(prediction: np.ndarray) -> np.ndarray:
    """
    Squeeze the prediction array to remove single-dimensional entries.

    Args:
        prediction (np.ndarray): The input prediction array.

    Returns:
        np.ndarray: The squeezed prediction array.
    """
    prediction = np.asarray(prediction)
    
    if prediction.ndim == 4:
        if prediction.shape[0] != 1:
            raise ValueError("Expected batch size of 1 for 4D prediction array")
        prediction = prediction[0]
    
    return prediction


def prediction_to_mask(
    prediction: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    task: str = "segmentation",
) -> np.ndarray:
    """
    Convert the model prediction to a binary mask based on the specified threshold.
    
    Args:
        prediction (np.ndarray): The model prediction array.
        threshold (float): The threshold value for converting probabilities to binary mask.
        task (str): The type of segmentation task. Currently supports "segmentation".
    
    Returns:
        np.ndarray: The binary mask generated from the prediction.
    """
    prediction = squeeze_prediction(prediction)
    
    if prediction.ndim == 2:
        prob = sigmoid(prediction)
        return (prob > threshold).astype(np.uint8)
    
    if prediction.ndim != 3:
        raise ValueError("Unsupported prediction shape for mask generation")
    
    channel_first = prediction.shape[0] < 32
    
    if not channel_first:
        prediction = np.transpose(prediction, (2, 0, 1))
    
    channel_count = prediction.shape[0]
    
    if task == "segmentation":
        if channel_count != 1:
            raise ValueError("Expected single-channel prediction for segmentation")
        
        prob = sigmoid(prediction[0])
        return (prob > threshold).astype(np.uint8)
    
    raise ValueError("Unsupported segmentation task.")


def remove_letterbox_padding(
    mask: np.ndarray,
    letterbox_meta: LetterboxMeta,
) -> np.ndarray:
    """
    Remove letterbox padding from the mask based on the provided metadata.

    Args:
        mask (np.ndarray): The input mask with letterbox padding.
        letterbox_meta (LetterboxMeta): Metadata containing original and resized sizes, and padding information.

    Returns:
        np.ndarray: The mask with letterbox padding removed.
    """
    resized_h, resized_w = letterbox_meta.resized_size
    pad_top = letterbox_meta.pad_top
    pad_left = letterbox_meta.pad_left
    
    return mask[
        pad_top:pad_top + resized_h, 
        pad_left:pad_left + resized_w,
    ]


def resize_mask_to_original(
    mask: np.ndarray,
    original_size: tuple[int, int],
) -> np.ndarray:
    """
    Resize the mask back to the original image size based on the provided metadata.

    Args:
        mask (np.ndarray): The input mask to be resized.
        original_size (tuple[int, int]): The size of the original image.

    Returns:
        np.ndarray: The resized mask matching the original image size.
    """
    original_h, original_w = original_size
    
    return cv2.resize(
        mask,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    )


def restore_mask_to_original(
    mask: np.ndarray,
    letterbox_meta: LetterboxMeta,
) -> np.ndarray:
    """
    Restore the mask to the original image size by removing letterbox padding and resizing.

    Args:
        mask (np.ndarray): The input mask with letterbox padding.
        letterbox_meta (LetterboxMeta): Metadata containing original and resized sizes, and padding information.

    Returns:
        np.ndarray: The restored mask matching the original image size.
    """
    unpadded_mask = remove_letterbox_padding(mask, letterbox_meta)
    return resize_mask_to_original(unpadded_mask, letterbox_meta.original_size)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Colorize the binary mask using the provided class colors.

    Args:
        mask (np.ndarray): The input binary mask.

    Returns:
        np.ndarray: The colorized mask.
    """
    if mask.ndim != 2:
        raise ValueError("Expected a 2D binary mask for colorization")
    
    colorized_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    for class_id, color in CLASS_COLORS_BGR.items():
        colorized_mask[mask == class_id] = color
    
    return colorized_mask


def overlay_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    alpha: float = DEFAULT_OVERLAY_ALPHA,
) -> np.ndarray:
    """
    Overlay the colorized mask onto the original image with specified transparency.

    Args:
        image_bgr (np.ndarray): The original BGR image.
        mask (np.ndarray): The binary mask to overlay.
        alpha (float): The transparency factor for the overlay.

    Returns:
        np.ndarray: The image with the mask overlay.
    """
    colorized_mask = colorize_mask(mask)
    
    blended = cv2.addWeighted(
        image_bgr,
        1.0 - alpha,
        colorized_mask,
        alpha,
        0,
    )
    
    overlay = image_bgr.copy()
    overlay[mask > 0] = blended[mask > 0]
    
    return overlay


def save_mask(mask_path: str | Path, mask: np.ndarray):
    """
    Save the binary mask to the specified path.

    Args:
        mask_path (str | Path): The path to save the mask.
        mask (np.ndarray): The binary mask to save.
    """
    mask_to_save = (mask).astype(np.uint8) * 255
    save_image(mask_path, mask_to_save)


def save_overlay(overlay_path: str | Path, overlay: np.ndarray):
    """
    Save the overlay image to the specified path.

    Args:
        overlay_path (str | Path): The path to save the overlay image.
        overlay (np.ndarray): The overlay image to save.
    """
    save_image(overlay_path, overlay)


def postprocess_prediction(
    prediction: np.ndarray,
    original_image_bgr: np.ndarray,
    letterbox: LetterboxMeta,
    threshold: float = DEFAULT_THRESHOLD,
    task: str = "segmentation",
    alpha: float = DEFAULT_OVERLAY_ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Postprocess the model prediction to generate a binary mask and overlay it on the original image.

    Args:
        prediction (np.ndarray): The model prediction array.
        original_image_bgr (np.ndarray): The original BGR image.
        letterbox (LetterboxMeta): Metadata containing original and resized sizes, and padding information.
        threshold (float): The threshold value for converting probabilities to binary mask.
        task (str): The type of segmentation task. Currently supports "segmentation".
        alpha (float): The transparency factor for the overlay.

    Returns:
        tuple[np.ndarray, np.ndarray]: The binary mask and the overlay image.
    """
    letterbox_mask = prediction_to_mask(prediction, threshold, task)
    restored_mask = restore_mask_to_original(letterbox_mask, letterbox)
    overlay = overlay_mask(original_image_bgr, restored_mask, alpha)
    
    return restored_mask, overlay