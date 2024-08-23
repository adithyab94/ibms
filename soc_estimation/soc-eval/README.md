# SOC Prediction of a Lithium-Ion Battery

## Getting started

This repository contains the code for the SOC prediction of a Lithium-Ion Battery. The code is written in Python 3.7 and uses Tensorflow 2.0. The code is written in a modular fashion and can be used to train and evaluate different models for the same task.

## Prerequisites

The minimum requirements required to run the project or specified as follows:

- Windows WSL2 or Windows Native (7 or higher)
- NVIDIA® GPU card with CUDA® Architecture
- Python 3.9
- cudatoolkit=11.2 cudnn=8.1.0
- Tensorflow ( <= 2.10 for windows native, 2.11 for wsl2)

A `.env` file is expected for hyperparameters and it should contain the following variable:

```python
PATIENCE=... # Early stopping patience
EPOCHS=... # Number of epochs
LEARNING_RATE=... # Learning rate
OUTPUT_MODEL = ... # Output model name
MODEL = .. # Model name to select

BATCH_SIZE=... # Batch size of tensorflow dataset DEFAULT=250
WINDOW_SIZE=... # Window size of tensorflow dataset DEFAULT=100
LABEL_SHIFT=... # Label shift of tensorflow dataset DEFAULT=1
```

## Setup

To Install Tensorflow with GPU support, Please follow the following steps: [Install Tensorflow](https://www.tensorflow.org/install)

Alternatively, you can use the `requirements.txt` provided in the repository to install the required packages

```bash
pip install -r requirements.txt
```

## Usage

The description of the API used in the project can be found [here](https://polarion.huber-group.com/polarion/#/project/iBMS/wiki/Huber%20Specification/AI%20Workflow)

To use the package, you can follow the below example:

Import the package

```python
import soc_eval
```

List datasets

```python
soc_eval.list_datasets()

out: ['lg', 'panasonic']
```

Load the LG Dataset

```python
soc_eval.load_dataset("lg")
```

Create a tensorflow dataset

```python
soc_eval.create_tf_dataset("lg")
```

Load the required model

```python
soc_eval.load_model('LSTM')

out: Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (1024, 100, 128)          67584     
                                                                 
 dropout (Dropout)           (1024, 100, 128)          0         
                                                                 
 leaky_re_lu (LeakyReLU)     (1024, 100, 128)          0         
                                                                 
 dense (Dense)               (1024, 100, 1)            129       
                                                                 
 activation (Activation)     (1024, 100, 1)            0         
                                                                 
=================================================================
Total params: 67,713
Trainable params: 67,713
Non-trainable params: 0
_________________________________________________________________
```

Train the model

```python
soc_eval.train_model(model, train_tfds, val_tfds)
```

Load the trained model

```python
loaded_model = soc_eval.load_saved_model()
```

Predict the SOC

```python
soc_eval.predict_soc(loaded_model, test_tfds)
```

Evaluate the model

```python
soc_eval.evaluate_model(loaded_model, test_tfds)
```
