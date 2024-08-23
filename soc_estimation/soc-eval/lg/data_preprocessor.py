# ----------------------------------------------------------------------------
#   Module to preprocess and prepare the LG dataset for training
# ----------------------------------------------------------------------------

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from functools import cached_property


import pandas as pd
import parameters as ps

import itertools

original_dir = Path(ps.RAW_DATA_DIR)
output_path = Path(ps.OUTPUT_DATA_DIR)
train_temps = ps.TRAIN_TEMPS
test_temps = ps.TEST_TEMPS
val_temps = ps.VAL_TEMPS

train_drive_cycles = ps.TRAIN_DRIVE_CYCLES
val_drive_cycles = ps.VAL_DRIVE_CYCLES
test_drive_cycles = ps.TEST_DRIVE_CYCLES


def get_dataset(
    path: str, temperature: str, drive_cycles: list, charge=True, resample=False
) -> pd.DataFrame:
    """
    Input:
    path: str,
    temperature: str,
    drive_cycles: list,
    charge=True,
    resample = False

    Output: pd.DataFrame

    Function to pre processdata
    1. Get corresponding charge and discharge files
    2. Upsample or downsample data
    """

    df_list = []
    folder_path = original_dir / temperature
    charge_file_list = list(folder_path.glob("*Charge*.csv"))
    discharge_file_list = [
        list(folder_path.glob(f"*{cycle}*.csv")) for cycle in drive_cycles
    ]
    discharge_files_flat = list(itertools.chain(*discharge_file_list))
    file_list = charge_file_list + discharge_files_flat
    file_dict = {}
    for file in file_list:
        df = pd.read_csv(file, parse_dates=True, skiprows=30)
        if df.shape[1] != 15:
            df.columns = [
                "Time Stamp",
                "Step",
                "Status",
                "Prog Time",
                "Step Time",
                "Cycle",
                "Cycle Level",
                "Procedure",
                "Voltage",
                "Current",
                "Temperature",
                "Capacity",
                "WhAccu",
                "Cnt",
            ]
        else:
            df.columns = [
                "Time Stamp",
                "Step",
                "Status",
                "Prog Time",
                "Step Time",
                "Cycle",
                "Cycle Level",
                "Procedure",
                "Voltage",
                "Current",
                "Temperature",
                "Capacity",
                "WhAccu",
                "Cnt",
                "Empty",
            ]
        df[["Time Stamp"]] = df[["Time Stamp"]].apply(pd.to_datetime)
        df = df.set_index("Time Stamp")
        df["data"] = temperature
        file_dict[file.name] = df

    if charge:
        od = sorted(file_dict, key=lambda x: file_dict[x].first_valid_index())
        filtered = od.copy()
        charge_corresponding = [
            filtered[filtered.index(x) + 1]
            for i in discharge_files_flat
            for x in filtered
            if filtered.index(x) + 1 < len(filtered)
            if i.name == x
        ]
        charge_filtered = [file for file in charge_corresponding if "Charge" in file]
        dataset_files = charge_filtered + [file.name for file in discharge_files_flat]

        for file in charge_filtered:
            df = file_dict[file]
            df = df[df["Status"].isin(["CHA", "DCH", "TABLE"])]
            df[["Capacity", "Voltage", "Current", "Batt_Temp"]] = df[
                ["Capacity", "Voltage", "Current", "Temperature"]
            ].apply(pd.to_numeric)
            max_charge = abs(max(df["Capacity"], default=0))
            df["SOC"] = (df["Capacity"]) / max_charge
            if resample:
                df = df[~df.index.duplicated(keep="first")]
                df = df.resample("1S").interpolate()
            df_list.append(df)

    for file in discharge_files_flat:
        df = file_dict[file.name]
        df = df[df["Status"].isin(["CHA", "DCH", "TABLE"])]
        df[["Capacity", "Voltage", "Current", "Batt_Temp"]] = df[
            ["Capacity", "Voltage", "Current", "Temperature"]
        ].apply(pd.to_numeric)
        max_discharge = abs(min(df["Capacity"]))  # , default=3
        df["SOC"] = (df["Capacity"] + max_discharge) / max_discharge
        if resample:
            df = df.resample("1S").first().interpolate()
        df_list.append(df)

    return pd.concat(df_list).sort_index()


def get_normalized_df(df) -> pd.DataFrame:
    """
    Input: pd.DataFrame raw data
    Output: normalized pd.DataFrame

    returns normalized dataframe
    """

    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(df[["Voltage", "Current", "Batt_Temp", "SOC"]])
    df[["Voltage", "Current", "Batt_Temp", "SOC"]] = normalized
    return df


@cached_property
def create_combined_dataset() -> pd.DataFrame:
    """
    Input: None
    Output: Filtered and concatenated dataFrame with only required columns

    Concatenate and filter test, train and validation data
    """
    train_df = pd.concat(
        [
            get_dataset(
                original_dir, temp, train_drive_cycles, charge=True, resample=False
            )
            for temp in train_temps
        ]
    )
    train_df["data"] = "train"
    train_df = train_df.fillna(0).sort_index()
    val_df = get_dataset(original_dir, "n10degC", val_drive_cycles, resample=True)
    val_df["data"] = "val"
    val_df = val_df.fillna(0).sort_index()
    test_df_0 = get_dataset(original_dir, "0degC", test_drive_cycles, resample=True)
    test_df_0 = test_df_0.sort_index()
    test_df_0["data"] = "test0"
    test_df_0 = test_df_0.fillna(0).sort_index()
    test_df_10 = get_dataset(original_dir, "10degC", test_drive_cycles, resample=True)
    test_df_10 = test_df_10.sort_index()
    test_df_10["data"] = "test10"
    test_df_10 = test_df_10.fillna(0).sort_index()
    test_df_25 = get_dataset(original_dir, "25degC", test_drive_cycles, resample=True)
    test_df_25["data"] = "test25"
    test_df_25 = test_df_25.fillna(0).sort_index()

    concat_df = pd.concat([train_df, test_df_0, test_df_10, test_df_25, val_df])
    concat_df = concat_df.sort_index().ffill()
    df_norm = get_normalized_df(concat_df)
    df_filtered = df_norm[["Voltage", "Current", "Batt_Temp", "SOC", "data"]]
    return df_filtered


def get_normalized_train_data() -> pd.DataFrame:
    """
    Input: None
    Output: Train data

    This function returns the train data
    """
    df_filtered = create_combined_dataset()
    train_df_norm = df_filtered[df_filtered["data"] == "train"]
    train_df_norm = train_df_norm[
        ["Voltage", "Current", "Batt_Temp", "SOC"]
    ].sort_index()
    return train_df_norm


def get_normalized_test_data(temperature: str) -> pd.DataFrame:
    """
    Input: temperature
    Output: Test data according to temperature

    This function returns the test data according to the temperature
    """
    df_filtered = create_combined_dataset()
    if temperature == "0degC":
        test_df_norm = df_filtered[df_filtered["data"] == "test0"]
        test_df_norm = test_df_norm[
            ["Voltage", "Current", "Batt_Temp", "SOC"]
        ].sort_index()
    elif temperature == "10degC":
        test_df_norm = df_filtered[df_filtered["data"] == "test10"]
        test_df_norm = test_df_norm[
            ["Voltage", "Current", "Batt_Temp", "SOC"]
        ].sort_index()
    elif temperature == "25degC":
        test_df_norm = df_filtered[df_filtered["data"] == "test25"]
        test_df_norm = test_df_norm[
            ["Voltage", "Current", "Batt_Temp", "SOC"]
        ].sort_index()
    return test_df_norm


def get_normalized_val_data() -> pd.DataFrame:
    """
    Input: None
    Output: Validation data

    returns normalized validation data
    """
    df_filtered = create_combined_dataset()
    val_df_norm = df_filtered[df_filtered["data"] == "val"]
    val_df_norm = val_df_norm[["Voltage", "Current", "Batt_Temp", "SOC"]].sort_index()
    return val_df_norm


def save_dataset(output_path):
    """
    Input: output_path
    Output: None

    saves dataset to the output path
    """
    get_normalized_train_data.to_csv(output_path / "train.csv")
    get_normalized_test_data("0degC").to_csv(output_path / "test0.csv")
    get_normalized_test_data("10degC").to_csv(output_path / "test10.csv")
    get_normalized_test_data("25degC").to_csv(output_path / "test25.csv")
    get_normalized_val_data.to_csv(output_path / "val.csv")
