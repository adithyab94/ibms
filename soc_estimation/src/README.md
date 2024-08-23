# SOC Estimation of a lithium-ion battery using tf-lite micro

The objective of this project is to convert the LSTM model to tensorflow lite for microcontroller format and deploy it to the ESP32-Ethernet-Kit to estimate state of charge of a lithium battery

## Installation

Follow detailed instructions provided specifically for this example.

Select the instructions depending on Espressif chip installed on your development board:

- [ESP32 Getting Started Guide](https://docs.espressif.com/projects/esp-idf/en/stable/get-started/index.html)
- [ESP32-S2 Getting Started Guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s2/get-started/index.html)

## Folder contents

ESP-IDF projects are built using CMake. The project build configuration is contained in `CMakeLists.txt` files that provide set of directives and instructions describing the project's source files and targets (executable, library, or both).

Below is short explanation of remaining files in the project folder.

```folder
├── CMakeLists.txt             Main CMake file for the project
├── components
│   ├── esp-nn
│   ├── tflite-lib
├── pytest_hello_world.py      Python script used for automated testing
├── main                       Main folder of the project
│   ├── CMakeLists.txt
│   └── input_arr.h             Test data for input array
│   └── main_functions.cc       Main functions of the project
|   └── main_functions.h        Header file for main functions
|   └── main.cc                 Main file of the project
|   └── model.cc                File with converted tf-lite model
|   └── model.h                 Header file for model.cc
|   └── output_handler.cc       File to handle output
|   └── output_handler.h        Header file for output_handler.cc       
├── sdkconfig                   Default configuration for the project       
└── README.md                   This is the file you are currently reading
```

## Deploy to ESP32

The following instructions will help you build and deploy this sample
to [ESP32](https://www.espressif.com/en/products/hardware/esp32/overview)
devices using the [ESP IDF](https://github.com/espressif/esp-idf).

The sample has been tested on ESP-IDF version `release/v4.2` and `release/v4.4` with the following devices:

- [ESP32-DevKitC](http://esp-idf.readthedocs.io/en/latest/get-started/get-started-devkitc.html)
- [ESP32-S3-DevKitC](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/hw-reference/esp32s3/user-guide-devkitc-1.html)
- [ESP-EYE](https://github.com/espressif/esp-who/blob/master/docs/en/get-started/ESP-EYE_Getting_Started_Guide.md)

### Install the ESP IDF

Follow the instructions of the
[ESP-IDF get started guide](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html)
to setup the toolchain and the ESP-IDF itself.

The next steps assume that the
[IDF environment variables are set](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html#step-4-set-up-the-environment-variables) :

- The `IDF_PATH` environment variable is set

- `idf.py` and Xtensa-esp32 tools (e.g. `xtensa-esp32-elf-gcc`) are in `$PATH`

### Building the example

Set the chip target (For esp32s3 target, IDF version `release/v4.4` is needed):

```bash
idf.py set-target esp32s3
```

Then build with `idf.py`

```bash
idf.py build
```

### Load and run the example

To flash (replace `/dev/ttyUSB0` with the device serial port):

```bash
idf.py --port /dev/ttyUSB0 flash
```

Monitor the serial output:

```bash
idf.py --port /dev/ttyUSB0 monitor
```

Use `Ctrl+]` to exit.
The previous two commands can be combined:

```bash
idf.py --port /dev/ttyUSB0 flash monitor
```
