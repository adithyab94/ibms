# ----------------------------------------------------------------------------
# Module to define the parameters and constants
# ----------------------------------------------------------------------------
from dotenv import load_dotenv
import os

load_dotenv()

# Data paths
DATASET_DIR = "datasets"
RAW_DATA_DIR = "datasets\\lg\\LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020"
OUTPUT_DATA_DIR = "datasets\\lg\\test\\"
OUTPUT_MODEL_DIR = "models\\"

# Temperature for training, validation and test
TRAIN_TEMPS = ["n10degC", "0degC", "10degC", "25degC"]
TEST_TEMPS = ["0degC", "10degC", "25degC"]
VAL_TEMPS = ["n10degC"]


TRAIN_DRIVE_CYCLES = ["Mixed"]
VAL_DRIVE_CYCLES = ["UDDS", "LA92", "US06"]
TEST_DRIVE_CYCLES = ["UDDS", "LA92", "US06"]

# Tensorflow dataset parameters
BATCH_SIZE = int(str(os.getenv("BATCH_SIZE")))
WINDOW_SIZE = int(str(os.getenv("WINDOW_SIZE")))
LABEL_SHIFT = int(str(os.getenv("LABEL_SHIFT")))

# Hyperparameters
PATIENCE = int(str(os.getenv("PATIENCE")))
EPOCHS = int(str(os.getenv("EPOCHS")))
LEARNING_RATE = float(str(os.getenv("LEARNING_RATE")))
OUTPUT_MODEL = os.getenv("OUTPUT_MODEL")
MODEL = os.getenv("MODEL")
