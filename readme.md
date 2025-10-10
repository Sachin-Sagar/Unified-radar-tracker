Unified Real-Time Radar Tracker
1. Project Summary
This project unifies two distinct Python systems into a single, real-time radar tracking application. It combines the hardware communication and data parsing capabilities of python_checkpoint3 with the advanced tracking algorithms of python-radar-tracker.

The application performs an end-to-end process:

Radar Configuration: It configures a connected radar sensor by sending a series of commands from a profile file.

Real-Time Data Acquisition: It listens to the serial port for incoming binary data frames from the radar.

Live Data Processing: For each frame, it parses the data, adapts it to the required format, and processes it through a sophisticated tracking pipeline.

Advanced Tracking: It uses an Interacting Multiple Model (IMM) filter combined with a Joint Probabilistic Data Association (JPDA) algorithm to robustly track multiple objects in real-time, even in cluttered environments. It also performs ego-motion estimation to distinguish between moving and stationary objects.

Data Logging: Upon closing the application (with Ctrl+C), it saves the complete history of processed frames and tracks into both a JSON file (for visualization) and a MATLAB-compatible .mat file (for debugging and validation).

The system is designed to be robust, with fallbacks for scenarios where supplementary data (like vehicle CAN bus information) is unavailable.

2. Folder Structure
unified_radar_tracker/
|
├── configs/
|   └── profile_80_m_40mpsec_bsdevm_16tracks_dyClutter.cfg
|
├── output/
|   └── (This directory will contain the final .json and .mat log files)
|
├── src/
|   ├── __init__.py
|   ├── data_adapter.py
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
3. File Descriptions
configs/
profile_...cfg: The configuration file sent to the radar on startup to define its operational parameters.

src/
data_adapter.py: The crucial bridge that translates FrameData objects from the hardware layer into the FHistFrame format expected by the tracking layer.

main_live.py: The main entry point for the application. It initializes the hardware, starts the tracker, and runs the main real-time processing loop.

src/hardware/
hw_comms_utils.py: Manages serial port communication, including opening the port, changing baud rates, and reliably finding the start of data frames.

parsing_utils.py: Handles parsing of the text-based radar configuration file and provides helper functions for unpacking binary data.

read_and_parse_frame.py: Decodes the raw binary data stream from the radar into a structured FrameData object containing the point cloud, targets, and stats.

src/tracking/
tracker.py: Defines the RadarTracker class, which encapsulates the entire state and logic of the tracking system, allowing it to process data one frame at a time.

perform_track_assignment_master.py: The central orchestrator for the tracking logic. For each frame, it calls the various track management functions in a strict, hierarchical order.

data_loader.py: Utility for loading and synchronizing pre-recorded .mat files, enabling the system to run in a post-processing mode for debugging.

export_to_json.py: Formats the final track and frame history into a specific JSON schema suitable for downstream visualization tools.

update_and_save_history.py: Saves the final processed data to both JSON and MATLAB-compatible .mat files upon application shutdown.

visualize_track_history.py: A standalone debugging tool to load a saved JSON file and interactively visualize the history of a single track.

src/tracking/algorithms/
my_dbscan.py: Implements a custom DBSCAN clustering algorithm to group raw radar points into object detections based on position and velocity.

estimate_ego_motion_ransac.py: A RANSAC-based algorithm to robustly estimate the vehicle's own velocity using only the radar point cloud.

estimate_ego_motion.py: Fuses radar, IMU, and CAN data in an EKF to produce a high-quality estimate of the vehicle's motion and identify moving points.

And other core algorithms for classification, barrier detection, and hypothesis generation.

src/tracking/filters/
imm_filter.py & imm_models.py: The Interacting Multiple Model (IMM) filter logic, which combines several motion models (Constant Velocity, Turn, Acceleration) for robust object tracking.

ego_ekf.py: The Extended Kalman Filter used for estimating the ego vehicle's motion.

src/tracking/track_management/
Contains the high-level logic for a track's lifecycle:

assign.py: Creates new tracks from unassigned detections.

update_tentative.py: Manages new tracks until they are confirmed as stable.

jpda_assignment.py: Manages confirmed tracks using the robust JPDA algorithm.

reassign.py: Attempts to revive tracks that have been temporarily lost.

delete.py: Marks tracks as lost if they are not seen for several frames.

src/tracking/utils/
A collection of helper functions for math operations, coordinate transforms, TTC categorization, and more.