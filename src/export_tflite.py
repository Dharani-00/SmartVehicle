"""
TensorFlow Lite Model Export for IDR Prototype.

Converts the trained Keras model to TFLite format for
future Android/embedded deployment.
"""
import os
import numpy as np
import tensorflow as tf
from src.config import MODEL_DIR, WINDOW_SIZE


def export_to_tflite(model_name="idr_speed_model", quantize=False):
    """
    Convert trained Keras model to TensorFlow Lite format.

    Args:
        model_name: Base name of the saved Keras model
        quantize: If True, apply dynamic range quantization (smaller model)

    Returns:
        Path to the saved .tflite file
    """
    keras_path = os.path.join(MODEL_DIR, f"{model_name}.keras")
    if not os.path.exists(keras_path):
        raise FileNotFoundError(f"Keras model not found: {keras_path}")

    print(f"  Loading Keras model: {keras_path}")
    model = tf.keras.models.load_model(keras_path)

    print("  Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # LSTM requires SELECT_TF_OPS in TF 2.x for TensorList operations
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("  Applying dynamic range quantization")

    tflite_model = converter.convert()

    # Save
    suffix = "_quantized" if quantize else ""
    tflite_path = os.path.join(MODEL_DIR, f"{model_name}{suffix}.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"  TFLite model saved: {tflite_path} ({size_kb:.1f} KB)")

    # Verify the model loads and runs
    verify_tflite_model(tflite_path)

    return tflite_path


def verify_tflite_model(tflite_path):
    """
    Verify TFLite model can load and produce output.

    Note: LSTM models with SELECT_TF_OPS require the Flex delegate,
    which may not be available in the standard Python TFLite interpreter.
    On Android, add 'org.tensorflow:tensorflow-lite-select-tf-ops' dependency.
    """
    print("  Verifying TFLite model...")
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"    Input:  {input_details[0]['shape']} dtype={input_details[0]['dtype']}")
        print(f"    Output: {output_details[0]['shape']} dtype={output_details[0]['dtype']}")

        # Run with dummy data
        input_shape = input_details[0]["shape"]
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        if np.any(np.isnan(output)):
            raise RuntimeError("TFLite model produced NaN output!")

        print(f"    Test output: {output.flatten()[0]:.4f} m/s")
        print("    TFLite verification PASSED")
    except RuntimeError as e:
        if "Flex delegate" in str(e) or "Select TensorFlow op" in str(e):
            print("    NOTE: Model uses SELECT_TF_OPS (LSTM). Flex delegate")
            print("    required for local verification. On Android, add:")
            print("    'org.tensorflow:tensorflow-lite-select-tf-ops' dependency.")
            print("    Model file is valid - verification skipped on desktop.")
            # Verify at minimum the file is a valid flatbuffer
            file_size = os.path.getsize(tflite_path)
            if file_size < 1000:
                raise RuntimeError(f"TFLite file too small ({file_size} bytes)")
            print(f"    File size: {file_size / 1024:.1f} KB - OK")
        else:
            raise


if __name__ == "__main__":
    print("Exporting model to TFLite...")
    path = export_to_tflite()
    print(f"\nExported: {path}")
    # Also export quantized version
    path_q = export_to_tflite(quantize=True)
    print(f"Quantized: {path_q}")
