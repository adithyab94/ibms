# ----------------------------------------------------------------------------
# Module to predict the SOC for the battery dataset
# ----------------------------------------------------------------------------


from matplotlib import pyplot as plt
from pathlib import Path

import lstm_models as lm
import tensorflow as tf
import parameters as ps
import lg.utils as lg
import pandas as pd


EPOCHS = ps.EPOCHS
tf_dataset = tf.data.Dataset


def list_datasets():
    """
    Input:  None
    Output: List of datasets

    Returns a list of datasets in the datasets folder
    """

    datasets_list = []
    for path in Path(ps.DATASET_DIR).glob("*"):
        datasets_list.append(path.name)
    return datasets_list


def load_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Input:
        dataset_name: Name of the dataset to be loaded
    Output:
        dataset: dataframe

    This function loads the dataset
    """

    if dataset_name == "lg":
        (
            train_df,
            test_df_0degC,
            test_df_10degC,
            test_df_25degC,
            test_df_n10degC,
            val_df,
        ) = lg.read_data()
        return (
            train_df,
            test_df_0degC,
            test_df_10degC,
            test_df_25degC,
            test_df_n10degC,
            val_df,
        )


def create_tf_dataset(dataset_name: str) -> tf_dataset:
    """
    Input:
        dataset_name: Name of the dataset
    Output:
        tf_dataset: tf dataset

    This function creates a tf dataset from a dataframe
    """

    if dataset_name == "lg":
        (
            train_df,
            test_df_0degC,
            test_df_10degC,
            test_df_25degC,
            test_df_n10degC,
            val_df,
        ) = load_dataset(dataset_name)
        train_tfds = lg.sequential_window_dataset(
            train_df,
            batch_size=ps.BATCH_SIZE,
            window_size=ps.WINDOW_SIZE,
            label_shift=ps.LABEL_SHIFT,
        )
        test_tfds_0degC = lg.sequential_window_dataset(
            test_df_0degC,
            batch_size=ps.BATCH_SIZE,
            window_size=ps.WINDOW_SIZE,
            label_shift=ps.LABEL_SHIFT,
        )
        test_tfds_10degC = lg.sequential_window_dataset(
            test_df_10degC,
            batch_size=ps.BATCH_SIZE,
            window_size=ps.WINDOW_SIZE,
            label_shift=ps.LABEL_SHIFT,
        )
        test_tfds_25degC = lg.sequential_window_dataset(
            test_df_25degC,
            batch_size=ps.BATCH_SIZE,
            window_size=ps.WINDOW_SIZE,
            label_shift=ps.LABEL_SHIFT,
        )
        test_tfds_n10degC = lg.sequential_window_dataset(
            test_df_n10degC,
            batch_size=ps.BATCH_SIZE,
            window_size=ps.WINDOW_SIZE,
            label_shift=ps.LABEL_SHIFT,
        )
        val_tfds = lg.sequential_window_dataset(
            val_df,
            batch_size=ps.BATCH_SIZE,
            window_size=ps.WINDOW_SIZE,
            label_shift=ps.LABEL_SHIFT,
        )
        return (
            train_tfds,
            test_tfds_0degC,
            test_tfds_10degC,
            test_tfds_25degC,
            test_tfds_n10degC,
            val_tfds,
        )


def list_models():
    """
    Input:  None
    Output: List of models

    Returns a list of models in the lstm_models.py
    """

    return ["LSTM", "Stacked_LSTM"]


def select_model(model_name) -> tf.keras.Model:
    """
    Input: model_name
    Output: Model

    This function returns the Selected LSTM model
    """
    if model_name == "LSTM":
        return lm.get_LSTM()
    elif model_name == "Stacked_LSTM":
        return lm.get_stacked_LSTM()
    else:
        print("Please select a valid model name")
        return None


def train_model(
    model: str, train_tfds: tf_dataset, val_tfds: tf_dataset
) -> tf.keras.Model:
    """
    Input:
        model: Model to be trained
        train_tfds: Training dataset
        test_tfds: Test dataset
        val_tfds: Validation dataset
    Output:
        Trained model

    This function trains the model and saves the model with the lowest validation loss
    """

    print(model.summary())
    print(ps.PATIENCE)

    history = model.fit(
        train_tfds,
        epochs=ps.EPOCHS,
        validation_data=val_tfds,
        callbacks=[
            lm.early_stopping,
            lm.model_checkpoint,
            lm.reset_states,
            lm.tensorboard_callback,
        ],
    )

    return history


def load_saved_model(model_name = ps.OUTPUT_MODEL) -> tf.keras.Model:
    """
    Input:
        model_name: Name of the model to be loaded
    Output:
        Loaded model

    This function loads the saved model
    """
    model = Path(ps.OUTPUT_MODEL_DIR) / model_name
    return tf.keras.models.load_model(
        model, custom_objects={"clipped_relu": lm.clipped_relu}
    )


def evaluate_model(model: tf.keras.Model, test_tfds: tf_dataset) -> None:
    """
    Input:
        model: Trained model
        test_tfds: Test dataset
    Output:
        None

    This function evaluates the model on the test dataset
    """

    loss, mae, rmse = model.evaluate(test_tfds, verbose=2)
    return loss, mae, rmse


def predict_model(model: tf.keras.Model, test_tfds: tf_dataset) -> None:
    """
    Input:
        model: Trained model
        test_tfds: Test dataset
    Output:
        None

    This function predicts the model on the test dataset
    """
    predictions = model.predict(test_tfds)
    return predictions


def plot_history(history):
    """
    Input:
        history: History of the training
    Output:
        None

    This function plots the training and validation loss
    """
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(len(loss))

    plt.figure()
    plt.plot(epochs, loss, "r", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()


def convert_to_tf_lite(model: tf.keras.Model, output_model: str) -> None:
    """
    Input:
        model: Trained model
        output_model: Path to save the model
    Output:
        None

    This function converts the model to tf lite format
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(output_model, "wb") as f:
        f.write(tflite_model)


def hex_to_C_array(input_file: str, output_file: str) -> None:
    """
    Input:
        input_file: Path to the input file
        output_file: Path to the output file
    Output:
        None

    This function converts the hex file to c array
    """
    with open(input_file, "rb") as f:
        data = f.read()
    data = [hex(x) for x in data]
    data = [x.replace("0x", "0x") for x in data]
    with open(output_file, "w") as f:
        f.write("unsigned char model[] = {")
        f.write(",".join(data))
        f.write("};")
