# Ip0355

## SOC Estimation of a lithium-ion battery using tf-lite micro

The objective of this project is to convert the LSTM model to tensorflow lite for microcontroller format and deploy it to the ESP32-Ethernet-Kit to estimate state of charge of a lithium battery

## Folder contents

Below is short explanation of remaining files in the project folder. A detailed information about the installation and instruction can be found in the README file of each folder.

```folder
├── doc             Documentation of the project
├── soc-eval        Python api to train,test the model and convert it to tensorflow lite for microcontroller format
├── src             Code to deploy the model to the ESP32-Ethernet-Kit
├── .gitignore      Git ignore file       
└── README.md       This is the file you are currently reading
```

## Description

The python api to train and test the model can be found in the `soc-eval` folder. A detailed information about the installation and instruction can be found in the ``soc-eval`'s [README](soc-eval/README.md) file.

The code to convert the model to tensorflow lite for microcontroller format can be found in the [`converter.py`](converter.py) file. To convert the LSTM model to tensorflow lite for microcontroller format, replace the following parameters in [`converter.py`](soc-eval/converter.py)

The code to deploy the model to the ESP32-Ethernet-Kit can be found in the `src` folder. A detailed information about the installation and instruction can be found in the `src`'s [README](src/README.md) file.

## Usage

### converter.py

To convert the LSTM model to tensorflow lite for microcontroller format, replace the following parameters in converter.py

```python
model_path = ... # Path to the model.h5 file
output_type = ...  # 'int8' or 'float16'
output_file_name = ... # Name of the output file
```

Then run the following command:

```bash
python converter.py
```
