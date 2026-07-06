import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    ALLOWED_IMAGE_EXTENSIONS,
    DEFAULT_INPUT_SIZE,
    DEFAULT_OVERLAY_ALPHA,
    DEFAULT_THRESHOLD,
    INPUT_IMAGE_DIR,
    MASK_FILENAME_SUFFIX,
    MODEL_DIR,
    OUTPUT_IMAGE_DIR,
    OVERLAY_FILENAME_SUFFIX,
)
from src.backend import create_backend  # noqa: E402
from src.postprocess import postprocess_prediction, save_mask, save_overlay  # noqa: E402
from src.preprocess import preprocess_image  # noqa: E402
from src.utils import ensure_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark EfficientUNet inference on an image directory."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True, choices=["onnx", "tflite"])
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", default=str(OUTPUT_IMAGE_DIR / "benchmark50"))
    parser.add_argument("--height", type=int, default=DEFAULT_INPUT_SIZE[0])
    parser.add_argument("--width", type=int, default=DEFAULT_INPUT_SIZE[1])
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--alpha", type=float, default=DEFAULT_OVERLAY_ALPHA)
    parser.add_argument("--task", default="segmentation", choices=["segmentation"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--save-outputs", action="store_true")
    return parser.parse_args()


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def validate_model_path(model_path: str | Path, backend: str) -> Path:
    path = resolve_project_path(model_path)
    model_root = MODEL_DIR.resolve()

    try:
        path.relative_to(model_root)
    except ValueError as exc:
        raise ValueError("Model must be inside the models directory.") from exc

    if not path.is_file():
        raise FileNotFoundError("Model file does not exist.")

    if backend == "onnx" and path.suffix.lower() != ".onnx":
        raise ValueError("ONNX backend requires a .onnx model.")

    if backend == "tflite" and path.suffix.lower() != ".tflite":
        raise ValueError("TFLite backend requires a .tflite model.")

    return path


def validate_image_dir(image_dir: str | Path) -> Path:
    path = resolve_project_path(image_dir)
    input_root = INPUT_IMAGE_DIR.resolve()

    try:
        path.relative_to(input_root)
    except ValueError as exc:
        raise ValueError("Image directory must be inside images/input.") from exc

    if not path.is_dir():
        raise NotADirectoryError("Image directory does not exist.")

    return path


def validate_output_dir(output_dir: str | Path) -> Path:
    path = resolve_project_path(output_dir)
    output_root = OUTPUT_IMAGE_DIR.resolve()

    try:
        path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError("Output directory must be inside images/output.") from exc

    return ensure_dir(path)


def list_images(image_dir: Path) -> list[Path]:
    images = [
        path
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    ]

    if not images:
        raise FileNotFoundError("No input images found.")

    return images


def class_name_from_file(image_path: Path) -> str:
    if "__" in image_path.stem:
        return image_path.stem.split("__", 1)[0]
    return "unknown"


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p90_index = max(0, int(len(ordered) * 0.9) - 1)

    return {
        "mean_ms": round(statistics.mean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
        "p90_ms": round(ordered[p90_index], 3),
    }


def save_prediction_outputs(
    output_dir: Path,
    image_path: Path,
    mask: Any,
    overlay: Any,
) -> None:
    mask_dir = ensure_dir(output_dir / "mask")
    overlay_dir = ensure_dir(output_dir / "overlay")

    mask_path = mask_dir / f"{image_path.stem}{MASK_FILENAME_SUFFIX}"
    overlay_path = overlay_dir / f"{image_path.stem}{OVERLAY_FILENAME_SUFFIX}"

    save_mask(mask_path=mask_path, mask=mask)
    save_overlay(overlay_path=overlay_path, overlay=overlay)


def run_warmup(
    backend: Any,
    image_path: Path,
    input_size: tuple[int, int],
    warmup_runs: int,
) -> None:
    if warmup_runs <= 0:
        return

    preprocessed = preprocess_image(
        image_path=image_path,
        input_size=input_size,
    )

    for _ in range(warmup_runs):
        _ = backend.predict(preprocessed.tensor)


def benchmark_image(
    backend: Any,
    image_path: Path,
    input_size: tuple[int, int],
    output_dir: Path,
    threshold: float,
    alpha: float,
    task: str,
    save_outputs: bool,
) -> dict[str, Any]:
    total_start = time.perf_counter()

    preprocess_start = time.perf_counter()
    preprocessed = preprocess_image(
        image_path=image_path,
        input_size=input_size,
    )
    preprocess_ms = elapsed_ms(preprocess_start)

    backend_start = time.perf_counter()
    prediction = backend.predict(preprocessed.tensor)
    backend_ms = elapsed_ms(backend_start)

    postprocess_start = time.perf_counter()
    mask, overlay = postprocess_prediction(
        prediction=prediction,
        original_image_bgr=preprocessed.original_bgr,
        letterbox=preprocessed.letterbox,
        threshold=threshold,
        task=task,
        alpha=alpha,
    )
    postprocess_ms = elapsed_ms(postprocess_start)

    save_ms = 0.0
    if save_outputs:
        save_start = time.perf_counter()
        save_prediction_outputs(output_dir, image_path, mask, overlay)
        save_ms = elapsed_ms(save_start)

    total_ms = elapsed_ms(total_start)

    return {
        "image": image_path.name,
        "class": class_name_from_file(image_path),
        "preprocess_ms": round(preprocess_ms, 3),
        "backend_ms": round(backend_ms, 3),
        "postprocess_ms": round(postprocess_ms, 3),
        "save_ms": round(save_ms, 3),
        "total_ms": round(total_ms, 3),
    }


def build_report(
    rows: list[dict[str, Any]],
    backend: str,
    model_path: Path,
    model_load_ms: float,
    benchmark_total_ms: float,
) -> dict[str, Any]:
    backend_values = [row["backend_ms"] for row in rows]
    total_values = [row["total_ms"] for row in rows]

    per_class: dict[str, dict[str, Any]] = {}
    classes = sorted({row["class"] for row in rows})

    for class_name in classes:
        class_rows = [row for row in rows if row["class"] == class_name]
        per_class[class_name] = {
            "count": len(class_rows),
            "backend": summarize([row["backend_ms"] for row in class_rows]),
            "total": summarize([row["total_ms"] for row in class_rows]),
        }

    return {
        "backend": backend,
        "model_file": model_path.name,
        "image_count": len(rows),
        "model_load_ms": round(model_load_ms, 3),
        "benchmark_total_ms": round(benchmark_total_ms, 3),
        "average_backend_ms": round(statistics.mean(backend_values), 3),
        "average_total_ms": round(statistics.mean(total_values), 3),
        "backend_summary": summarize(backend_values),
        "total_summary": summarize(total_values),
        "per_class": per_class,
        "per_image": rows,
    }


def write_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "image",
        "class",
        "preprocess_ms",
        "backend_ms",
        "postprocess_ms",
        "save_ms",
        "total_ms",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(json_path: Path, report: dict[str, Any]) -> None:
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


def main() -> None:
    args = parse_args()

    model_path = validate_model_path(args.model, args.backend)
    image_dir = validate_image_dir(args.image_dir)
    output_dir = validate_output_dir(args.output)
    input_size = (args.height, args.width)
    image_paths = list_images(image_dir)

    load_start = time.perf_counter()
    backend = create_backend(model_path=model_path, backend=args.backend)
    model_load_ms = elapsed_ms(load_start)

    run_warmup(
        backend=backend,
        image_path=image_paths[0],
        input_size=input_size,
        warmup_runs=args.warmup,
    )

    benchmark_start = time.perf_counter()
    rows = [
        benchmark_image(
            backend=backend,
            image_path=image_path,
            input_size=input_size,
            output_dir=output_dir,
            threshold=args.threshold,
            alpha=args.alpha,
            task=args.task,
            save_outputs=args.save_outputs,
        )
        for image_path in image_paths
    ]
    benchmark_total_ms = elapsed_ms(benchmark_start)

    report = build_report(
        rows=rows,
        backend=args.backend,
        model_path=model_path,
        model_load_ms=model_load_ms,
        benchmark_total_ms=benchmark_total_ms,
    )

    csv_path = output_dir / "benchmark_results.csv"
    json_path = output_dir / "benchmark_report.json"

    write_csv(csv_path, rows)
    write_json(json_path, report)

    print("Benchmark completed.")
    print(f"Images: {report['image_count']}")
    print(f"Model load: {report['model_load_ms']:.3f} ms")
    print(f"Total benchmark time: {report['benchmark_total_ms']:.3f} ms")
    print(f"Average backend inference: {report['average_backend_ms']:.3f} ms/image")
    print(f"Average total pipeline: {report['average_total_ms']:.3f} ms/image")
    print(f"CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"JSON: {json_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()