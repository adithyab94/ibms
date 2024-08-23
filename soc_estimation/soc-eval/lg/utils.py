# ----------------------------------------------------------------------------
#  Module to define necessary utils and functions for the LG Dataset
# ----------------------------------------------------------------------------

from matplotlib import pyplot as plt
from functools import cached_property
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt
from pathlib import Path

import parameters as ps
import plotly.graph_objs as go

import tensorflow as tf
import pandas as pd


DATA_PATH = Path(ps.OUTPUT_DATA_DIR)
tf_dataset = tf.data.Dataset


def read_data():
    """
    Input:
        DATA_PATH: Path to the data directory
    Output:
        train_df: Training dataframe
        test_df_25degC : Test dataset at 25 degC
        test_df_0degC : Test dataset at 10 degC
        test_df_0degc : Test dataset at 0 degC
        test_df_n10degC : Test dataset at -10 degC
        val_df : Validation dataset
    """
    train_df = pd.read_csv(
        DATA_PATH / "train.csv", index_col="Time Stamp", parse_dates=True
    )
    test_df_0degC = pd.read_csv(
        DATA_PATH / "test0.csv", index_col="Time Stamp", parse_dates=True
    )
    test_df_10degC = pd.read_csv(
        DATA_PATH / "test10.csv", index_col="Time Stamp", parse_dates=True
    )
    test_df_25degC = pd.read_csv(
        DATA_PATH / "test25.csv", index_col="Time Stamp", parse_dates=True
    )
    test_df_n10degC = pd.read_csv(
        DATA_PATH / "val.csv", index_col="Time Stamp", parse_dates=True
    )
    val_df = pd.read_csv(
        DATA_PATH / "val.csv", index_col="Time Stamp", parse_dates=True
    )
    return (
        train_df,
        test_df_0degC,
        test_df_10degC,
        test_df_25degC,
        test_df_n10degC,
        val_df,
    )


def sequential_window_dataset(
    dataframe: pd.DataFrame, batch_size=250, window_size=100, label_shift=1
) -> tf_dataset:
    """
    Input:
        dataframe: Dataframe to be converted into a tf dataset
        batch_size: Batch size for the dataset
        window_size: Size of the window to be used for the dataset
        label_shift: Shift of the label with respect to the features
    Output:
        dataset: tf dataset

    This function creates a tf dataset from a dataframe. The dataset is created by splitting
    the dataframe into windows of size window_size and
    then converting the windows into a tf dataset.
    """

    def split(window):
        return window[:, :-1], window[:, -1, tf.newaxis]

    dataset = tf.data.Dataset.from_tensor_slices(
        dataframe
    )  # Transformation of dataset into tensor slices. Creates a Dataset whose elements are slices of the given tensors.
    dataset = dataset.window(
        window_size, shift=label_shift, stride=1, drop_remainder=True
    )
    dataset = dataset.flat_map(
        lambda window: window.batch(window_size)
    )  # Maps map_func across this dataset and flattens the result. Use flat_map if you want to make sure that the order of your dataset stays the same. For example, to flatten a dataset of batches into a dataset of their elements:
    dataset = dataset.map(
        split, num_parallel_calls=tf.data.AUTOTUNE
    )  # map_func can accept as arguments and return any type of dataset element. num_parallel_calls representing the number of batches to compute asynchronously in parallel
    dataset = (
        dataset.cache()
    )  # Caches the elements in this dataset.The first time the dataset is iterated over, its elements will be cached either in the specified file or in memory. Subsequent iterations will use the cached
    dataset = dataset.batch(
        batch_size=batch_size, drop_remainder=True
    )  # Combines consecutive elements of this dataset into batches
    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )  # Preparing the immediate next batch to run the input data processing more faster
    return dataset


def get_tf_dataset() -> tf_dataset:
    """
    Input:
        train_df: Training dataframe
        test_df_25degC : Test dataset at 25 degC
        test_df_0degC : Test dataset at 10 degC
        test_df_0degc : Test dataset at 0 degC
        test_df_n10degC : Test dataset at -10 degC
        val_df: Validation dataframe

    Output:
        train_tfds: Training tf dataset
        test_tfds_25degc : Test tf dataset at 25 degC
        test_tfds_10degc : Test tf dataset at 10 degC
        test_tfds_0degc : Test tf dataset at 0 degC
        test_tfds_n10degc : Test tf dataset at -10 degC
        val_tfds: Validation tf dataset

    This function creates tf datasets from the dataframes
    """
    (
        train_df,
        test_df_0degC,
        test_df_10degC,
        test_df_25degC,
        test_df_n10degC,
        val_df,
    ) = read_data(DATA_PATH)
    train_tfds = sequential_window_dataset(train_df)
    test_tfds_25degc = sequential_window_dataset(test_df_25degC)
    test_tfds_10degc = sequential_window_dataset(test_df_10degC)
    test_tfds_0degc = sequential_window_dataset(test_df_0degC)
    test_tfds_n10degc = sequential_window_dataset(test_df_n10degC)
    val_tfds = sequential_window_dataset(val_df)
    return (
        train_tfds,
        test_tfds_25degc,
        test_tfds_10degc,
        test_tfds_0degc,
        test_tfds_n10degc,
        val_tfds,
    )


def plot_predictions(predictions, test_df: pd.DataFrame) -> None:
    """
    Input:
        model: Trained model
        test_tfds: Test dataset
    Output:
        None

    This function plots the predictions of the model
    """

    test_pred = pd.DataFrame(
        predictions
    )  # converting the predicted values it into a dataframe
    test_pred = test_pred.rename(
        columns={0: "SOC_predicted"}, index={"ONE": "Row_1"}
    )  # renaming the dataframe
    test_label = test_df["SOC_Observed"].iloc[
        0 : len(test_pred)
    ]  # preparing the length of dataframe
    test_pred_obs = [test_label, test_pred]
    test_pred_obs = pd.concat(test_pred_obs, axis=1)

    error = (
        (test_pred_obs["SOC_predicted"] - test_pred_obs["SOC_Observed"])
        / test_pred_obs["SOC_Observed"]
    ) * 100  # Percentage error calculation
    error = pd.DataFrame(error)
    error = error.rename(columns={0: "Percentage_Error"}, index={"ONE": "Row_1"})

    # Percentage error with Moving average filter
    error_moving_avg = error["Percentage_Error"].rolling(window=10).mean()

    # Differencce error calculation
    diff_error = test_pred_obs["SOC_predicted"] - test_pred_obs["SOC_Observed"]
    diff_error = pd.DataFrame(diff_error)
    diff_error = diff_error.rename(
        columns={0: "Difference_Error"}, index={"ONE": "Row_1"}
    )

    # Create figure with secondary y-axis
    fig_testing = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig_testing.add_trace(
        go.Scatter(x=test_pred.index, y=test_pred["SOC_predicted"], name="Predicted"),
        secondary_y=False,
    )
    fig_testing.add_trace(
        go.Scatter(x=test_label.index, y=test_label, name="Observed"),
        secondary_y="Secondary",
    )
    # Add figure title
    fig_testing.update_layout(
        title_text="<b>Observed versus Predicted SOC at 25degC for Testing Dataset</b>"
    )
    # Set x-axis title
    fig_testing.update_xaxes(title_text="Time(s)")
    # Set y-axes titles
    fig_testing.update_yaxes(title_text="<b>primary</b> Predicted", secondary_y=False)
    fig_testing.update_yaxes(title_text="<b>secondary</b> Observed", secondary_y=True)
    fig_testing.show()

    # Create figure with secondary y-axis
    fig_error = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig_error.add_trace(
        go.Scatter(
            x=error_moving_avg.index, y=error_moving_avg, name="Percentage Error"
        ),
        secondary_y=False,
    )
    fig_error.add_trace(
        go.Scatter(
            x=error_moving_avg.index,
            y=diff_error["Difference_Error"],
            name="Difference Error",
        ),
        secondary_y="Difference Error",
    )
    # Add figure title
    fig_error.update_layout(
        title_text="<b>Percentage Error and Difference Error at 25degC for Testing Dataset</b>"
    )
    # Set x-axis title
    fig_error.update_xaxes(title_text="Time(s)")
    # Set y-axes titles
    fig_error.update_yaxes(
        title_text="<b>primary</b> Percentage Error", secondary_y=False
    )
    fig_error.update_yaxes(
        title_text="<b>secondary</b> Difference Error", secondary_y=True
    )
    fig_error.show()
