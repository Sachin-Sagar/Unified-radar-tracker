# src/data_adapter.py

import numpy as np
from .hardware.read_and_parse_frame import FrameData

class FHistFrame:
    """A class to mimic the structure of a single fHist frame from the .mat file."""
    def __init__(self):
        self.timestamp = 0.0
        self.pointCloud = np.array([])
        self.posLocal = np.array([])
        # --- Initialize other fields used by the tracker with default values ---
        self.motionState = 0
        self.isOutlier = np.array([], dtype=bool)
        self.egoVx = 0.0
        self.egoVy = 0.0
        self.correctedEgoSpeed_mps = 0.0
        self.estimatedAcceleration_mps2 = np.nan
        self.iirFilteredVx_ransac = 0.0
        self.iirFilteredVy_ransac = 0.0
        self.grid_map = []
        self.dbscanClusters = np.array([])
        self.detectedClusterInfo = np.array([])
        self.filtered_barrier_x = None # Will be populated by the tracker

def adapt_frame_data_to_fhist(frame_data, last_timestamp_ms):
    """
    Converts a real-time FrameData object into an FHistFrame object
    that the tracking algorithms can process.

    Args:
        frame_data (FrameData): The parsed data from the radar.
        last_timestamp_ms (float): The timestamp of the previous frame in ms.

    Returns:
        FHistFrame: An object with the same structure as a single fHist frame.
    """
    fhist_frame = FHistFrame()

    # 1. Map Timestamp
    # The tracker expects timestamp in milliseconds. 
    # The original MATLAB script assumes a 50ms frame time. We will do the same.
    # A more robust solution could use actual system time.
    fhist_frame.timestamp = last_timestamp_ms + 50.0

    # 2. Map Point Cloud and Position Data
    # The tracker expects pointCloud: [range, x, y, doppler, snr]
    # and posLocal: [x, y]
    if frame_data.point_cloud is not None and frame_data.point_cloud.size > 0:
        fhist_frame.pointCloud = frame_data.point_cloud
        # Extract the x, y coordinates for posLocal
        fhist_frame.posLocal = frame_data.point_cloud[1:3, :]
    else:
        # Ensure arrays are empty but have the correct dimensions if there are no points
        fhist_frame.pointCloud = np.empty((5, 0))
        fhist_frame.posLocal = np.empty((2, 0))

    # 3. Initialize isOutlier array based on number of points
    fhist_frame.isOutlier = np.zeros(frame_data.num_points, dtype=bool)

    # The other fields (like egoVx, motionState, etc.) will be calculated
    # by the tracking logic itself, so we just need to ensure they exist.

    return fhist_frame