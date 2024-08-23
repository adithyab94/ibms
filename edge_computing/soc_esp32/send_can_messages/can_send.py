import pickle
import pandas as pd
import cantools
import can
import time
import struct

def read_pkl_file(file_path):
    """
    Reads a .pkl file and returns the data.
    
    Parameters:
        file_path (str): The path to the .pkl file.
    
    Returns:
        data: The data read from the .pkl file.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def extract_relevant_data(data):
    """
    Extracts the first three columns (voltage, current, temperature) from the data.
    
    Parameters:
        data: The data from the .pkl file.
    
    Returns:
        pd.DataFrame: A DataFrame containing only the first three columns.
    """
    # Assuming data is a numpy array, convert it to a DataFrame for ease of use
    df = pd.DataFrame(data, columns=["Voltage", "Current", "Batt_Temp"])
    return df[["Voltage", "Current", "Batt_Temp"]]

def float_to_bytes(value):
    """
    Converts a float to bytes.
    
    Parameters:
        value (float): The floating-point value.
    
    Returns:
        bytes: The byte representation of the float.
    """
    return struct.pack('f', value)

def send_can_messages(data, dbc_file_path, can_channel, bitrate):
    """
    Sends CAN messages based on the data using the DBC file for message definitions.
    
    Parameters:
        data (pd.DataFrame): The data containing voltage, current, temperature.
        dbc_file_path (str): The path to the DBC file.
        can_channel (str): The CAN channel to use (e.g., 'PCAN_USBBUS1').
        bitrate (int): The bitrate for the CAN bus.
    """
    # Load DBC file
    db = cantools.database.load_file(dbc_file_path)

    # Set up CAN bus with specified bitrate
    try:
        bus = can.interface.Bus(channel=can_channel, bustype='pcan', bitrate=bitrate)
    except Exception as e:
        print(f"An error occurred while setting up the CAN bus: {e}")
        return

    for index, row in data.iterrows():
        voltage_bytes = float_to_bytes(row["Voltage"])
        current_bytes = float_to_bytes(row["Current"])
        temperature_bytes = float_to_bytes(row["Batt_Temp"])

        # Prepare CAN message payload
        can_data = bytearray(8)

        # Assuming the voltage, current, and temperature are to be sent in different messages
        can_data[0:4] = voltage_bytes
        can_data[4:8] = current_bytes  # Adjust start and end indices as needed

        # Send Voltage and Current message
        try:
            current_message = db.get_message_by_frame_id(0x119)
            current_msg = can.Message(arbitration_id=0x119, data=can_data, is_extended_id=False)
            bus.send(current_msg)
            print(f"Sent Voltage and Current message {index+1}/{len(data)}")
        except Exception as e:
            print(f"Error encoding/sending Voltage and Current message: {e}")

        # Prepare CAN message payload for Temperature
        can_data[0:4] = temperature_bytes
        can_data[4:8] = bytearray(4)  # Clear the rest of the data if not used

        # Send Temperature message
        try:
            temp_message_1_2 = db.get_message_by_frame_id(0x133)
            temp_msg_1_2 = can.Message(arbitration_id=0x133, data=can_data, is_extended_id=False)
            bus.send(temp_msg_1_2)
            print(f"Sent Temp1,2 message {index+1}/{len(data)}")
        except Exception as e:
            print(f"Error encoding/sending Temperature message: {e}")

        # Add a delay to prevent the transmit queue from filling up
        time.sleep(0.01)  # 10 ms delay

if __name__ == "__main__":
    file_path = 'data/test_x_0deg.pkl'  # Replace with your actual file path
    dbc_file_path = 'signals.dbc'  # Replace with your actual DBC file path
    can_channel = 'PCAN_USBBUS1'  # Replace with your actual CAN channel
    bitrate = 500000  # Set the CAN bus bitrate to 500 kbps

    data = read_pkl_file(file_path)
    
    if data is not None:
        relevant_data = extract_relevant_data(data)
        send_can_messages(relevant_data, dbc_file_path, can_channel, bitrate)
