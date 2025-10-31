# Unified Real-Time Radar Tracker

## Project Overview

This project is a complete, real-time radar tracking application built in Python. It interfaces directly with a radar sensor, parses incoming binary data, and processes it through an advanced tracking pipeline. The application can be run in a live mode with hardware or in a playback mode using pre-recorded data files for testing and validation.

The system is designed to be robust, featuring a sophisticated sensor fusion and clutter rejection architecture to accurately track objects in a variety of conditions.

**Key Technologies:**

*   **Programming Language:** Python
*   **Core Libraries:**
    *   `numpy`: For numerical operations and array manipulation.
    *   `pyserial`: For serial communication with the radar hardware.
    *   `PyQt5`: For the GUI and live visualizer.
    *   `pyqtgraph`: For real-time plotting.
    *   `scipy`: For scientific and technical computing.
    *   `psutil`: For monitoring system resources.

**Architecture:**

*   **Dual-Mode Operation:** The application can run in "Live Mode" (with hardware) or "Playback Mode" (from `.mat` log files).
*   **Hardware Interfacing:** The application sends configuration commands to the radar sensor over a serial port.
*   **Data Processing:**
    *   Incoming binary data is parsed into a usable format.
    *   An Interacting Multiple Model (IMM) filter and a Joint Probabilistic Data Association (JPDA) algorithm are used for robust object tracking.
    *   Ego-motion is estimated using RANSAC to differentiate between moving and stationary objects.
    *   A dual-box filtering system rejects stationary clutter.
*   **Data Logging:** Raw and processed data are saved to JSON files, and console output is logged to both text and JSON files.

## Building and Running

1.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure COM Port (for Live Mode):**

    Open `src/main_live.py` and adjust the `CLI_COMPORT_NUM` variable to match the serial port your radar is connected to. For Linux, the correct path is `/dev/ttyACM0`.

3.  **Run the Application:**

    ```bash
    python main.py
    ```

4.  **Select a Mode:**

    The script will prompt you to choose between:
    1.  Live Tracking
    2.  Playback from File

5.  **View Results:**

    Log files (raw data, track history, and console output) are saved in the `output/` directory.

## Development Conventions

*   **Modular Structure:** The codebase is organized into modules with specific responsibilities (e.g., `hardware`, `tracking`, `filters`).
*   **Configuration:** Radar configuration is managed through `.cfg` files in the `configs/` directory.
*   **Debugging:** A `debug_mode` flag in `src/tracking/parameters.py` can be enabled for verbose tracking messages.
*   **Data Flow:** The `data_adapter.py` module acts as a bridge between the hardware/playback layers and the tracking layer, ensuring a consistent data format.
*   **Visualization:** A standalone debugging tool (`visualize_track_history.py`) is available to load and visualize saved track histories.
