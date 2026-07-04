import argparse
from pathlib import Path

from config import (
    DEFAULT_INPUT_SIZE,
    DEFAULT_OVERLAY_ALPHA,
    DEFAULT_THRESHOLD,
    MASK_FILENAME_SUFFIX,
    MODEL_DIR,
    OUTPUT_IMAGE_DIR,
    OVERLAY_FILENAME_SUFFIX,
)
from src.backend import create_backend
from src.preprocess import preprocess_image
from src.postprocess import postprocess_prediction, save_mask, save_overlay
from src.utils import Timer, ensure_dir, get_logger, safe_filename


def parse_args():
    parser = argparse.ArgumentParser(
        description="EfficientUNet semantic segmentation inference"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default=str(OUTPUT_IMAGE_DIR))
    parser.add_argument("--backend", required=True, choices=["onnx", "tflite"])
    parser.add_argument("--height", type=int, default=DEFAULT_INPUT_SIZE[0])
    parser.add_argument("--width", type=int, default=DEFAULT_INPUT_SIZE[1])
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--alpha", type=float, default=DEFAULT_OVERLAY_ALPHA)
    parser.add_argument("--task", default="binary", choices=["binary", "multiclass"])
    return parser.parse_args()


def validate_model_path(model_path: str | Path, backend: str) -> Path:
    path = Path(model_path).resolve()
    model_root = MODEL_DIR.resolve()
    
    try:
        path.relative_to(model_root)
    except ValueError as e:
        raise ValueError("Model must be inside the models directory.") from e
    
    if not path.is_file():
        raise FileNotFoundError("Model file does not exist.")
    
    file_extension = path.suffix.lower()
    
    if backend == "onnx" and file_extension != ".onnx":
        raise ValueError("ONNX backend requires a .onnx model")
    
    if backend == "tflite" and file_extension != ".tflite":
        raise ValueError("TFLite backend requires a .tflite model")
    
    return path


def validate_output_path(output_path: str | Path) -> Path:
    path = Path(output_path).resolve()
    output_root = OUTPUT_IMAGE_DIR.resolve()
    
    try:
        path.relative_to(output_root)
    except ValueError as e:
        raise ValueError("Output path must be inside images/output.")
    
    return ensure_dir(path)


def build_output_paths(output_dir: Path, image_path: str | Path) -> tuple[Path, Path]:
    image_stem = Path(image_path).stem
    
    mask_path = output_dir/f"{image_stem}{MASK_FILENAME_SUFFIX}"
    overlay_path = output_dir/f"{image_stem}{OVERLAY_FILENAME_SUFFIX}"
    
    return mask_path, overlay_path


def run_inference(args) -> tuple[Path, Path]:
    logger = get_logger("EfficientUNet")
    
    model_path = validate_model_path(args.model_path, args.backend)
    output_dir = validate_output_path(args.output)
    input_size = (args.height, args.width)
    
    logger.info("Starting Inference")
    
    preprocessed = preprocess_image(
        image_path=args.image,
        input_size=input_size
    )
    
    backend = create_backend(
        model_path=model_path,
        backend=args.backend
    )
    
    with Timer() as timer:
        prediction = backend.predict(preprocessed.tensor)
    
    logger.info("Backend inference time: %.2f ms.", timer.elapsed_ms)
    
    mask, overlay = postprocess_prediction(
        prediction=prediction,
        original_image_bgr=preprocessed.original_bgr,
        letterbox=preprocessed.letterbox,
        threshold=args.threshold,
        task=args.task,
        alpha=args.alpha,
    )
    
    mask_path, overlay_path = build_output_paths(
        output_dir=output_dir,
        image_path=args.image
    )
    
    save_mask(mask_path=mask_path, mask=mask)
    save_overlay(overlay_path=overlay_path, overlay=overlay)
    
    logger.info("Saved mask: %s", safe_filename(mask_path))
    logger.info("Saved overlay: %s", safe_filename(overlay_path))
    
    return mask_path, overlay_path


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
