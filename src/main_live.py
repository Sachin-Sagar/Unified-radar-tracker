import sys
import time
from datetime import datetime
import numpy as np

# --- Import all our project modules ---
# Hardware layer
from .hardware import hw_comms_utils, parsing_utils
from .hardware.read_and_parse_frame import read_and_parse_frame

# Data adapter (the bridge)
from .data_adapter import adapt_frame_data_to_fhist

# Tracking layer
from .tracking.tracker import RadarTracker
from .tracking.main import define_parameters # Use main.py from tracker to get params
from .tracking.update_and_save_history import update_and_save_history

# --- Configuration ---
if sys.platform == "win32":
    CLI_COMPORT_NUM = 'COM11' # <-- ADJUST THIS to your COM port number
elif sys.platform == "linux":
    CLI_COMPORT_NUM = '/dev/ttyACM0'
else:
    print(f"Unsupported OS '{sys.platform}' detected. Please set COM port manually.")
    CLI_COMPORT_NUM = None
    
CONFIG_FILE_PATH = 'configs/profile_80_m_40mpsec_bsdevm_16tracks_dyClutter.cfg'
INITIAL_BAUD_RATE = 115200

def configure_sensor_and_params(cli_com_port, chirp_cfg_file):
    """
    Reads the config file, sends commands to the radar, and returns
    the parsed parameters and the open data port.
    (Adapted from python_checkpoint3/main.py)
    """
    cli_cfg = parsing_utils.read_cfg(chirp_cfg_file)
    if not cli_cfg:
        return None, None

    params = parsing_utils.parse_cfg(cli_cfg)
    
    target_baud_rate = INITIAL_BAUD_RATE
    for command in cli_cfg:
        if command.startswith("baudRate"):
            try:
                target_baud_rate = int(command.split()[1])
            except (ValueError, IndexError):
                print(f"Warning: Could not parse baud rate from command: {command}")
            break

    print("\n--- Starting Sensor Configuration ---")
    h_data_port = hw_comms_utils.configure_control_port(cli_com_port, INITIAL_BAUD_RATE)
    if not h_data_port:
        return None, None
        
    for command in cli_cfg:
        print(f"> {command}")
        h_data_port.write((command + '\n').encode())
        time.sleep(0.1)
        
        if "baudRate" in command:
            time.sleep(0.2)
            try:
                h_data_port.baudrate = target_baud_rate
                print(f"  Baud rate changed to {target_baud_rate}")
            except Exception as e:
                print(f"ERROR: Failed to change baud rate: {e}")
                h_data_port.close()
                return None, None

    print("--- Configuration complete ---\n")
    hw_comms_utils.reconfigure_port_for_data(h_data_port)
    return params, h_data_port

def main():
    """Main application entry point."""
    # 1. Configure and start the radar sensor
    params_radar, h_data_port = configure_sensor_and_params(CLI_COMPORT_NUM, CONFIG_FILE_PATH)
    if not params_radar or not h_data_port:
        print("Failed to configure sensor. Exiting.")
        return

    # 2. Initialize the Tracker and Data History
    params_tracker = define_parameters()
    tracker = RadarTracker(params_tracker)
    fhist_history = [] # To store processed frames for saving later

    print("--- Starting Live Tracking (Press Ctrl+C to stop) ---")
    try:
        while True:
            # 3. Read and parse one frame from the radar
            frame_data = read_and_parse_frame(h_data_port, params_radar)
            if not frame_data or not frame_data.header:
                continue

            # 4. Adapt the data format from hardware to tracker
            fhist_frame = adapt_frame_data_to_fhist(frame_data, tracker.last_timestamp_ms)

            # 5. Process the frame with the tracker
            # For a true live system, you would get CAN data here and pass it in.
            # For now, we pass None to use the radar-only fallbacks.
            updated_tracks, processed_frame = tracker.process_frame(fhist_frame, can_signals=None)
            
            # 6. Store the processed frame for saving later
            fhist_history.append(processed_frame)

            # 7. (Optional) Print real-time results
            num_confirmed = sum(1 for t in updated_tracks if t.get('isConfirmed') and not t.get('isLost'))
            print(f"Frame: {tracker.frame_idx} | Detections: {frame_data.num_points} | Confirmed Tracks: {num_confirmed}")

    except KeyboardInterrupt:
        print("\n--- Stopping application and saving data ---")
        # 8. Save the final results when the program is stopped
        if tracker.all_tracks and fhist_history:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output/track_history_{timestamp}.json"
            update_and_save_history(
                tracker.all_tracks,
                fhist_history,
                filename,
                params=tracker.params
            )
        else:
            print("No data was processed, nothing to save.")

    except Exception as e:
        print(f"\n--- An unexpected error occurred: {e} ---")

    finally:
        if h_data_port and h_data_port.is_open:
            h_data_port.close()
            print("--- Serial port closed ---")

if __name__ == '__main__':
    main()