# ----------------------------------------------------------------------------
#  Module to define the LSTM models for the LG dataset
# ----------------------------------------------------------------------------

from tensorflow import keras
from pathlib import Path

import tensorflow as tf
import numpy as np
import parameters as ps

import datetime


keras.backend.clear_session()  # Clearing te backround running session
tf.random.set_seed(42)  # Random number initialization for weights and biases
np.random.seed(42)


PATIENCE = ps.PATIENCE
EPOCHS = ps.EPOCHS
OUTPUT_MODEL = ps.OUTPUT_MODEL
LEARNING_RATE = ps.LEARNING_RATE
OUTPUT_MODEL_DIR = Path(ps.OUTPUT_MODEL_DIR)

# Defining callbacks for the model


class ResetStatesCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs):
        self.model.reset_states()


reset_states = ResetStatesCallback()


def clipped_relu(x):
    """
    Custom activation function for the output layer
    """
    return tf.keras.activations.relu(x, max_value=1.0)


model_checkpoint = keras.callbacks.ModelCheckpoint(
    OUTPUT_MODEL_DIR / OUTPUT_MODEL, save_best_only=True
)


early_stopping = keras.callbacks.EarlyStopping(patience=PATIENCE)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    profile_batch="100,101",
)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: LEARNING_RATE * 0.5 ** (epoch / 10000)
)


lr = tf.keras.optimizers.schedules.ExponentialDecay(
    LEARNING_RATE, 10000, 0.5, staircase=False, name=None
)


def get_LSTM() -> keras.Model:
    """
    Input: None
    Output:
    Single LSTM layer model
    """

    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(
                64,
                activation="tanh",
                return_sequences=True,
                stateful=True,
                unroll=False,  # LSTM first layer with 10 hidden units
                batch_input_shape=[1024, 100, 3],
            ),  # input_shape: Batch size, timesteps(LSTM Cell Number), Number of Features
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32),
            tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Activation(clipped_relu),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=ps.LEARNING_RATE
        ),  # ADAM learning rate Optimizer
        loss=tf.keras.losses.MeanSquaredError(),  # Regression Loss function MeanSquaredError
        metrics=[
            "mae",
            tf.keras.metrics.RootMeanSquaredError(),
        ],  # Accuracy matrics for evaluating the performance of the model
    )
    return model


def get_stacked_LSTM() -> keras.Model:
    """
    Input: None
    Output: Stacked LSTM layer model
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(
                64,
                activation="tanh",
                return_sequences=True,
                stateful=True,
                unroll=False,  # LSTM first layer with 10 hidden units
                batch_input_shape=[ps.BATCH_SIZE, ps.WINDOW_SIZE, 3],
            ),  # input_shape: Batch size, timesteps(LSTM Cell Number), Number of Features
            tf.keras.layers.LSTM(
                64,
                activation="tanh",
                return_sequences=False,
                stateful=True,
                unroll=False,  # LSTM first layer with 10 hidden units
                batch_input_shape=[ps.BATCH_SIZE, ps.WINDOW_SIZE, 3],
            ),  # input_shape: Batch size, timesteps(LSTM Cell Number), Number of Features
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(
                32
            ),  # Fully connected 1 hidden unit with custom activation function
            tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.Dense(
                32
            ),  # Fully connected 1 hidden unit with custom activation function
            tf.keras.layers.LeakyReLU(alpha=0.3),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Activation(clipped_relu),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=ps.LEARNING_RATE
        ),  # ADAM learning rate Optimizer
        loss=tf.keras.losses.MeanSquaredError(),  # Regression Loss function MeanSquaredError
        metrics=[
            "mae",
            tf.keras.metrics.RootMeanSquaredError(),
        ],  # Accuracy matrics for evaluating the performance of the model
    )
    return model
