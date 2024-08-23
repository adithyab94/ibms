import tensorflow as tf
import numpy as np
import pandas as pd
from .lg.utils import sequential_window_dataset
from .lstm_models import clipped_relu

### Parameters

model_path = "seq_to_one_50.h5"
output_type = "int8"  # 'int8' or 'float16'
test_data_path = "./datasets/lg/split/test0.csv"
output_file_name = "test "


def representative_dataset_gen(test_data_path=test_data_path):
    """
    Input:
        test_data_path: Path to the test data
    Output:
        Yielded representative dataset

    This function generates the representative dataset for quantization
    """
    for sample in (
        sequential_window_dataset(test_data_path).batch(1).take(100)
    ):  # Take 100 samples for representative dataset
        input_features = sample[0].numpy()
        input_features = [np.array(input_features, dtype=np.float32)]
        yield input_features


def convert_to_tf_lite_int_8(input_file: str, output_file: str) -> None:
    """
    Input:
        input_file: Path to the input file
        output_file: Path to the output file
    Output:
        Saves the converted int8 model to the output file

    This function converts the tf lite model to int 8
    """

    converter = tf.lite.TFLiteConverter.from_keras_model(input_file)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset_gen
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_quant_model = converter.convert()
    with open(output_file, "wb") as f:
        f.write(tflite_quant_model)


def convert_to_tf_lite_float_16(input_file: str, output_file: str) -> None:
    """
    Input:
        input_file: Path to the input file
        output_file: Path to the output file
    Output:
        Saves the converted float16 model to the output file

    This function converts the tf lite model to float 16
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(input_file)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.representative_dataset = representative_dataset_gen
    tflite_quant_model = converter.convert()
    with open(output_file, "wb") as f:
        f.write(tflite_quant_model)


def convert_tflite_to_cpp(tflite_path: str, c_header_path: str, input_param_type: str):
    """
    Input:
        tflite_path: Path to the TFLite model
        c_header_path: Path to the C header file
        input_param_type: Type of the input parameter. Must be either int8 or float16.
    Output:
        Converts the TFLite model to a C header file and saves it to the c_header_path
    """
    # Load the TFLite model
    with open(tflite_path, "rb") as f:
        tflite_model = f.read()

    # Convert the TFLite model to a TensorFlow Lite Interpreter object
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Get input and output tensor details
    input_details = interpreter.get_input_details()

    # Get the input shape
    input_shape = input_details[0]["shape"]

    # Generate CC file
    with open(c_header_path, "w") as f:
        f.write("#ifndef MODEL_H_\n#define MODEL_H_\n\n")
        f.write(f"const unsigned int model_data_len = {len(tflite_model)};\n")
        f.write(f"const int input_tensor_shape[] = {{")
        for dim in input_shape:
            f.write(f"{dim}, ")
        f.write(f"}};\n")
        f.write(f'const char* input_param_type = "{input_param_type}";\n')
        f.write(f"const unsigned char model_data[] = {{\n")
        for byte in tflite_model:
            f.write(f"0x{byte:02x}, ")
        f.write(f"}};\n")
        f.write("#endif // MODEL_H_")


def main(model_path: str, output_type: str, output_file_name: str):
    """
    Input:
        model_path: Path to the H5 model
        output_type: Type of the output model: int8, float16
        output_file_name: Name of the output file
    Output:
        None

    This function is the main function
    """
    model = tf.keras.models.load_model(
        model_path, custom_objects={"clipped_relu": clipped_relu}
    )

    if output_type == "int8":
        convert_to_tf_lite_int_8(model, output_file_name + ".tflite")
    elif output_type == "float16":
        convert_to_tf_lite_float_16(model, output_file_name + ".tflite")
    else:
        raise ValueError(
            "Invalid input parameter type. Must be either int8 or float16."
        )
    convert_tflite_to_cpp(
        output_file_name + ".tflite", output_file_name + ".cc", output_type
    )


if __name__ == "__main__":
    main(model_path, output_type, output_file_name)
