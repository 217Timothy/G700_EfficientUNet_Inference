from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_ROOT / "models"
INPUT_IMAGE_DIR = PROJECT_ROOT / "images" / "input"
OUTPUT_IMAGE_DIR = PROJECT_ROOT / "images" / "output"

DEFAULT_MODEL_PATH = MODEL_DIR / "model.onnx"

DEFAULT_INPUT_SIZE = (512, 512)  # Default input size for the model (height, width)

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)

DEFAULT_THRESHOLD = 0.5
DEFAULT_OVERLAY_ALPHA = 0.45

MASK_FILENAME_SUFFIX = "_mask.png"
OVERLAY_FILENAME_SUFFIX = "_overlay.png"

CLASS_COLORS_BGR = {
    0: (0, 0, 0),        # background
    1: (0, 0, 255),      # class 1: red
}

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
MAX_IMAGE_PIXELS = 4096 * 4096