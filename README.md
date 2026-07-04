# EfficientUNet Inference for Genio 700

Runtime-only semantic segmentation inference project for a MediaTek Genio 700
EVK running Linux.

This repository is intended for deployment and inference only. It does not
contain training code, training data, model conversion scripts, or model
artifacts.

## Project Structure

```text
EfficientUNet_Inference/
├── README.md
├── config.py
├── requirements-evk.txt
├── run.sh
├── models/
│   └── .gitkeep
├── images/
│   ├── input/
│   │   └── .gitkeep
│   └── output/
│       └── .gitkeep
└── src/
    ├── backend.py
    ├── inference.py
    ├── postprocess.py
    ├── preprocess.py
    └── utils.py
```

Expected runtime artifacts are copied into the project after cloning:

```text
models/model.tflite
images/input/<input_image>
images/output/
```

Model files and image data are intentionally ignored by Git.

## Runtime Requirements

The EVK runtime environment should stay minimal:

```text
numpy
opencv-python-headless
tflite-runtime
```

Do not install training or conversion dependencies on the EVK, such as
PyTorch, TensorFlow, ONNX, ONNX Runtime, or onnx2tf, unless there is a specific
debugging reason.

## Installation on Genio 700 EVK

Clone the repository on the EVK:

```bash
git clone <repo-url>
cd EfficientUNet_Inference
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install runtime dependencies:

```bash
pip install --upgrade pip
pip install -r requirements-evk.txt
```

Copy the TFLite model into the runtime model directory:

```text
models/model.tflite
```

Copy input images into:

```text
images/input/
```

## Running Inference

Example:

```bash
python -m src.inference \
  --model models/model.tflite \
  --backend tflite \
  --image images/input/test0.png \
  --output images/output
```

The output directory will contain:

```text
images/output/mask/<image_name>_mask.png
images/output/overlay/<image_name>_overlay.png
```

Useful arguments:

```text
--model      Path to the model inside models/
--backend    Inference backend: tflite or onnx
--image      Input image path inside images/input/
--output     Output directory inside images/output/
--height     Model input height, default 512
--width      Model input width, default 512
--threshold  Segmentation threshold, default 0.5
--alpha      Overlay transparency, default 0.45
```

For Genio 700 deployment, use:

```text
--backend tflite
```

The ONNX backend is kept for local validation and backend compatibility work.

## Preprocessing

The preprocessing pipeline is implemented in `src/preprocess.py` and includes:

- Loading an input image from disk.
- Validating file extension and image size.
- Converting BGR to RGB.
- Letterbox resizing to the configured model input size.
- Normalizing with ImageNet mean and standard deviation.
- Converting HWC image layout to CHW tensor layout.
- Adding a batch dimension.

The inference tensor shape used by the shared pipeline is:

```text
[1, 3, height, width]
```

The TFLite backend handles layout conversion internally when the TFLite model
expects NHWC input:

```text
[1, height, width, 3]
```

## Postprocessing

The postprocessing pipeline is implemented in `src/postprocess.py` and includes:

- Numerically stable sigmoid for binary segmentation logits.
- Thresholding probabilities into a segmentation mask.
- Removing letterbox padding.
- Resizing the mask back to the original image size.
- Saving the binary mask.
- Saving an overlay image for visual inspection.

## Backend Design

Inference backend logic is isolated in `src/backend.py`.

Current backends:

```text
ONNXBackend
TFLiteBackend
```

`src/inference.py` does not contain backend-specific runtime details. It only:

1. Parses arguments.
2. Loads and preprocesses the image.
3. Creates the selected backend.
4. Runs prediction.
5. Postprocesses and saves results.

This keeps future backend changes localized to `src/backend.py`.

## Security and Repository Policy

This project may be used with confidential model and image artifacts. Keep the
repository clean:

- Do not commit `.pt`, `.pth`, `.onnx`, `.onnx.data`, `.tflite`, `.engine`, or
  checkpoint files.
- Do not commit input images or output masks.
- Do not commit local editor folders such as `.vscode/`.
- Do not commit virtual environments such as `.venv/`.
- Do not commit conversion scripts if they contain internal paths, model
  details, or environment-specific commands.
- Keep model transfer outside Git, for example by using SSH, SCP, or an
  approved internal file transfer process.

The `.gitignore` file is configured to keep local artifacts out of version
control. If a sensitive artifact was committed in the past, removing it from the
latest commit is not enough; Git history must also be cleaned.

## Model Files

The EVK runtime should normally use:

```text
models/model.tflite
```

If using ONNX locally for validation, keep related ONNX external data files next
to the ONNX model:

```text
models/model.onnx
models/model.onnx.data
```

Both files are required when the ONNX model uses external tensor data.

## Notes

- This project is inference-only.
- Model conversion should be performed in a separate trusted development
  environment.
- Runtime code should not require internet access.
- Runtime logs avoid printing full internal paths when possible.
