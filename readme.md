Unified Real-Time Radar Tracker
1. Project Summary
This project is a complete, real-time radar tracking application built in Python. It interfaces directly with a radar sensor, parses incoming binary data, and processes it through an advanced tracking pipeline. The application can be run in a live mode with hardware or in a playback mode using pre-recorded data files for testing and validation.

The system is designed to be robust, featuring a sophisticated sensor fusion and clutter rejection architecture to accurately track objects in a variety of conditions.

2. Features
Dual-Mode Operation: Can be launched in Live Mode to connect directly to radar hardware or in Playback Mode to process data from .mat log files.

Hardware Interfacing: Configures the radar sensor by sending commands from a profile file over a serial port.

Real-Time Data Acquisition: Listens to the serial port for incoming binary data frames, synchronizes them, and parses them into a usable format.

Live Visualization: Displays the radar's point cloud in real-time using PyQt5 and pyqtgraph, running in a separate thread to ensure a non-blocking UI.

Advanced Tracking:

Uses an Interacting Multiple Model (IMM) filter combined with a Joint Probabilistic Data Association (JPDA) algorithm to robustly track multiple objects.

Performs ego-motion estimation using RANSAC to distinguish between moving and stationary objects.

Implements a dual-box filtering system (static and dynamic) to reject stationary clutter from the environment (e.g., guardrails) while still tracking legitimate stationary targets (e.g., stopped vehicles).

Comprehensive Data Logging:

Saves the raw, unprocessed data from every frame to a radar_log_[timestamp].json file.

Saves the final processed frame and track history to a track_history_[timestamp].json file.

Saves all console output to both a human-readable text file and a structured JSON file for post-processing and debugging.

3. Folder Structure
unified_radar_tracker/
|
├── configs/
|   └── profile_80_m_40mpsec_bsdevm_16tracks_dyClutter.cfg
|
├── output/
|   └── (This directory will contain the final log files)
|
├── src/
|   ├── main_live.py
|   ├── main_playback.py
|   ├── ... (other modules)
|
├── main.py
└── readme.md
4. How to Run
Install Dependencies: Make sure you have the required Python libraries installed.

Bash

pip install numpy pyserial pyqt5 pyqtgraph scipy
Configure COM Port (for Live Mode): Open src/main_live.py and adjust the CLI_COMPORT_NUM variable to match the serial port your radar is connected to.

Run the Application: Execute the main.py script from the root directory of the project.

Bash

python main.py
Select a Mode: The script will prompt you to choose between (1) Live Tracking or (2) Playback from File.

(Optional) Enable Debug Logs: To see verbose tracking messages, open src/tracking/parameters.py and set the debug_mode flag to True.

View Results: When you close the application window, all log files (raw data, track history, and console output) will be saved in the output/ directory.

5. File Descriptions
Core Application Files
main.py: The main entry point for the application. It prompts the user to select between live and playback modes and launches the appropriate module.

main_live.py: Handles the setup and execution of the live tracking mode, including hardware configuration, starting the GUI, and managing the worker thread.

main_playback.py: Handles the setup and execution of the playback mode, including loading radar and CAN data from files and feeding it to the tracker.

live_visualizer.py: Creates the PyQt5 GUI window and the pyqtgraph plot for real-time data display.

console_logger.py: Configures the application-wide logging system to output messages to the console and to both text and JSON files.

json_logger.py: Implements a thread-safe DataLogger to save raw radar data to a JSON file without blocking the main processing loop.

data_adapter.py: A crucial bridge that translates data objects from both the hardware layer and .mat files into a consistent FHistFrame format expected by the tracking layer.

src/hardware/
hw_comms_utils.py: Manages serial port communication, including opening the port and reliably finding the start of data frames.

parsing_utils.py: Handles parsing of the radar configuration file and provides helper functions for unpacking binary data.

read_and_parse_frame.py: Decodes the raw binary data stream from the radar into a structured FrameData object.

src/tracking/
tracker.py: Defines the RadarTracker class, which encapsulates the entire state and logic of the tracking system. It orchestrates the pre-processing pipeline, including the dual-box (static and dynamic) logic for stationary clutter rejection.

perform_track_assignment_master.py: The central orchestrator for the tracking logic. For each frame, it executes the entire tracking pipeline in a hierarchical order: prediction, assignment (for confirmed, tentative, and new tracks), and deletion.

assign.py: Creates new tracks from unassigned detections. It uses strict filtering logic based on motion and location (the static box) to prevent creating tracks from environmental clutter.

jpda_assignment.py: Implements the Joint Probabilistic Data Association (JPDA) algorithm for confirmed tracks. It now uses robust, explicit matrix summation to prevent runtime errors from shape corruption.

imm_filter.py: Implements the Interacting Multiple Model (IMM) filter. The prediction step is now fortified to enforce correct matrix shapes, preventing the propagation of state corruption errors between frames.

update_and_save_history.py: Saves the final processed data to both JSON and MATLAB-compatible .mat files.

visualize_track_history.py: A standalone debugging tool to load a saved JSON file and interactively visualize the history of a single track.