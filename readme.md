Unified Real-Time Radar Tracker
1. Project Summary
This project is a complete, real-time radar tracking application built in Python. It interfaces directly with a radar sensor, configures it, parses the incoming binary data, and processes it through an advanced tracking pipeline. The application provides a live visualization of the radar's point cloud and saves all raw and processed data for later analysis.

The system is designed to be robust, with fallbacks for scenarios where supplementary data (like vehicle CAN bus information) is unavailable.

2. Features
Hardware Interfacing: Configures the radar sensor by sending commands from a profile file over a serial port.

Real-Time Data Acquisition: Listens to the serial port for incoming binary data frames and synchronizes them.

Live Visualization: Displays the radar's point cloud in real-time using PyQt5 and pyqtgraph, running in a separate thread to ensure a non-blocking UI.

Advanced Tracking:

Uses an Interacting Multiple Model (IMM) filter combined with a Joint Probabilistic Data Association (JPDA) algorithm to robustly track multiple objects.

Performs ego-motion estimation to distinguish between moving and stationary objects.

Comprehensive Data Logging:

Saves the raw, unprocessed data from every frame to a radar_log_[timestamp].json file.

Saves the final processed frame and track history to a track_history_[timestamp].json file for visualization and a MATLAB-compatible track_history_[timestamp]_python.mat file for debugging and validation.

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
|   ├── __init__.py
|   ├── data_adapter.py
|   ├── json_logger.py
|   ├── live_visualizer.py
|   ├── main_live.py
|   |
|   ├── hardware/
|   |   ├── __init__.py
|   |   ├── hw_comms_utils.py
|   |   ├── parsing_utils.py
|   |   └── read_and_parse_frame.py
|   |
|   └── tracking/
|       ├── __init__.py
|       ├── data_loader.py
|       ├── export_to_json.py
|       ├── perform_track_assignment_master.py
|       ├── tracker.py
|       ├── update_and_save_history.py
|       ├── visualize_track_history.py
|       |
|       ├── algorithms/
|       ├── filters/
|       ├── track_management/
|       └── utils/
|
└── readme.md
4. How to Run
Install Dependencies: Make sure you have the required Python libraries installed.

Bash

pip install numpy pyserial pyqt5 pyqtgraph scipy
Configure COM Port: Open src/main_live.py and adjust the CLI_COMPORT_NUM variable to match the serial port your radar is connected to.

Run the Application: Execute the main_live.py script from the root directory of the project.

Bash

python -m src.main_live
View Results: A live plot of the radar data will appear. When you close the window, the log files will be saved in the output/ directory.

5. File Descriptions
src/
main_live.py: The main entry point for the application. It initializes the GUI and starts the background worker thread for radar processing.

live_visualizer.py: Contains the LiveVisualizer class, which creates the PyQt5 GUI window and the pyqtgraph plot for real-time data display.

json_logger.py: Implements a thread-safe DataLogger class to save raw FrameData objects to a JSON file without blocking the main processing loop.

data_adapter.py: A crucial bridge that translates FrameData objects from the hardware layer into the FHistFrame format expected by the tracking layer.

src/hardware/
hw_comms_utils.py: Manages serial port communication, including opening the port, changing baud rates, and reliably finding the start of data frames.

parsing_utils.py: Handles parsing of the text-based radar configuration file and provides helper functions for unpacking binary data.

read_and_parse_frame.py: Decodes the raw binary data stream from the radar into a structured FrameData object containing the point cloud, targets, and stats.

src/tracking/
tracker.py: Defines the RadarTracker class, which encapsulates the entire state and logic of the tracking system.

perform_track_assignment_master.py: The central orchestrator for the tracking logic. For each frame, it calls the various track management functions in a strict, hierarchical order.

update_and_save_history.py: Saves the final processed data to both JSON and MATLAB-compatible .mat files upon application shutdown.

visualize_track_history.py: A standalone debugging tool to load a saved JSON file and interactively visualize the history of a single track.

src/tracking/algorithms/
my_dbscan.py: Implements a custom DBSCAN clustering algorithm to group raw radar points into object detections.

estimate_ego_motion.py: Fuses radar, IMU, and CAN data in an EKF to produce a high-quality estimate of the vehicle's motion.

...and other core algorithms for classification, barrier detection, and hypothesis generation.

src/tracking/filters/
imm_filter.py & imm_models.py: The Interacting Multiple Model (IMM) filter logic, which combines several motion models for robust object tracking.

ego_ekf.py: The Extended Kalman Filter used for estimating the ego vehicle's motion.

src/tracking/track_management/
Contains the high-level logic for a track's lifecycle (assigning, updating, reassigning, and deleting tracks).

src/tracking/utils/
A collection of helper functions for math operations, coordinate transforms, and more.