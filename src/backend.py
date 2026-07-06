from pathlib import Path
from typing import Protocol

import numpy as np


class InferenceBackend(Protocol):
    """
    Protocol for inference backends.
    """

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Perform inference on the input tensor.

        Args:
            input_tensor (np.ndarray): The input tensor for inference.

        Returns:
            np.ndarray: The output prediction from the model.
        """
        ...


class ONNXBackend:
    """
    Inference backend for ONNX models.
    """

    def __init__(self, model_path: str | Path):
        """
        Initialize the ONNX backend with the specified model path.

        Args:
            model_path (str | Path): The path to the ONNX model file.
        """
        import onnxruntime as ort
        
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Perform inference on the input tensor using the ONNX model.

        Args:
            input_tensor (np.ndarray): The input tensor for inference.

        Returns:
            np.ndarray: The output prediction from the model.
        """
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor.astype(np.float32)},
        )
        return outputs[0] # type: ignore


class TFLiteBackend:
    def __init__(self, model_path: str | Path) -> None:
        try:
            import ai_edge_litert.interpreter import Interpreter # type: ignore
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter # type: ignore
            except ImportError:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
        
        self.interpreter = Interpreter(model_path=str(model_path)) # type: ignore
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        if len(self.input_details) != 1:
            raise ValueError("TFLite model must have exactly one input.")

        if len(self.output_details) != 1:
            raise ValueError("TFLite model must have exactly one output.")

    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        input_index = self.input_details[0]["index"]
        output_index = self.output_details[0]["index"]
        input_dtype = self.input_details[0]["dtype"]
        expected_shape = self.input_details[0]["shape"]

        model_input = input_tensor

        if (
            model_input.ndim == 4
            and len(expected_shape) == 4
            and model_input.shape[1] == expected_shape[3]
        ):
            model_input = np.transpose(model_input, (0, 2, 3, 1))

        self.interpreter.set_tensor(
            input_index,
            model_input.astype(input_dtype),
        )
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(output_index)

        if output.ndim == 4 and output.shape[-1] < 32:
            output = np.transpose(output, (0, 3, 1, 2))

        return output


def create_backend(
    model_path: str | Path,
    backend: str,
) -> InferenceBackend:
    model_path = Path(model_path)

    if backend == "onnx":
        return ONNXBackend(model_path)

    if backend == "tflite":
        return TFLiteBackend(model_path)

    raise ValueError("Unsupported inference backend.")
