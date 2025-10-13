# src/tracking/tracker.py

import numpy as np
import numbers # <-- Import the numbers module for type checking

# Import all the necessary components from the tracking module
from .perform_track_assignment_master import perform_track_assignment_master
from .algorithms.estimate_ego_motion import estimate_ego_motion
from .algorithms.classify_vehicle_motion import classify_vehicle_motion
from .algorithms.detect_side_barrier import detect_side_barrier
from .algorithms.my_dbscan import my_dbscan
from .utils.slot_points_to_grid import slot_points_to_grid

class RadarTracker:
    def __init__(self, params):
        """Initializes the tracker's state."""
        self.params = params
        self.all_tracks = []
        self.next_track_id = 1
        self.frame_idx = 0
        self.last_timestamp_ms = 0.0

        # Initialize filter states
        self.ego_kf_state = {
            'x': np.zeros((5, 1)), 'P': np.diag([10.0, 10.0, 5.0, 5.0, 1.0]),
            'Q': np.diag([0.1, 0.1, 0.5, 0.5, 0.2]),
        }
        self.original_ego_kf_r = np.diag([0.2, 5.0, 1.0, 1.0, 0.3])
        self.filtered_vx_ego_iir = 0.0
        self.filtered_vy_ego_iir = 0.0
        self.right_turn_state = {}
        self.left_turn_state = {}
        
        default_x_range = self.params['barrier_detect_params']['default_x_range']
        self.filtered_barrier_x = {'left': default_x_range[0], 'right': default_x_range[1]}

    def _update_frame_with_can_data(self, frame, can_signals):
        """Updates the frame object with the interpolated CAN signals."""
        if can_signals:
            for signal_name, value in can_signals.items():
                setattr(frame, signal_name, value)
        return frame

    def process_frame(self, current_frame, can_signals=None):
        """
        Processes a single frame of radar data and updates the tracks.
        This contains the logic from the loop in the original tracker's main.py.
        """
        # Calculate delta_t
        delta_t = (current_frame.timestamp - self.last_timestamp_ms) / 1000.0 if self.frame_idx > 0 else 0.05
        if delta_t <= 0: delta_t = 0.05
        self.last_timestamp_ms = current_frame.timestamp

        # --- Prepare CAN and IMU data with robust fallbacks ---
        if can_signals:
            can_speed = can_signals.get('VehSpeed_Act_kmph', np.nan)
            can_torque = can_signals.get('ShaftTorque_Est_Nm', np.nan)
            can_gear = can_signals.get('Gear_Engaged_St_enum', np.nan)
            
            def get_numeric(value, default):
                if not isinstance(value, numbers.Number) or np.isnan(value):
                    return default
                return value

            can_grade = get_numeric(can_signals.get('EstimatedGrade_Est_Deg'), 0.0)
            imu_ax = get_numeric(can_signals.get('imuProc_xaccel'), 0.0)
            imu_ay = get_numeric(can_signals.get('imuProc_yaccel'), 0.0)
            imu_omega = get_numeric(can_signals.get('imuProc_yawRate'), 0.0)
        else:
            can_speed, can_torque, can_gear, can_grade = np.nan, np.nan, np.nan, 0.0
            imu_ax, imu_ay, imu_omega = 0.0, 0.0, 0.0
        
        current_frame = self._update_frame_with_can_data(current_frame, can_signals)

        cartesian_pos_data = current_frame.posLocal
        point_cloud = current_frame.pointCloud
        num_points = cartesian_pos_data.shape[1]

        # --- Pre-processing Block (Motion Classification, Ego-Motion, Barriers) ---
        current_time_s = current_frame.timestamp / 1000.0
        motion_state, self.right_turn_state, self.left_turn_state = classify_vehicle_motion(
            current_time_s, imu_omega,
            self.params['ego_motion_classification_params']['turn_yawRate_thrs'],
            self.params['ego_motion_classification_params']['confirmation_samples'],
            self.right_turn_state, self.left_turn_state
        )
        current_frame.motionState = motion_state

        (self.ego_kf_state, self.filtered_vx_ego_iir, self.filtered_vy_ego_iir,
         ransac_vx, ransac_vy, _, ax_dynamics, outlier_indices) = estimate_ego_motion(
             cartesian_pos_data.T, point_cloud[3, :] if num_points > 0 else np.array([]),
             can_speed, can_torque, can_gear, can_grade, imu_ax, imu_ay, imu_omega,
             np.deg2rad(can_grade), 0.0, self.ego_kf_state,
             self.filtered_vx_ego_iir, self.filtered_vy_ego_iir, delta_t,
             self.params['vehicle_params'], self.params['ego_motion_params'], self.original_ego_kf_r
         )
        
        if outlier_indices.size > 0: current_frame.isOutlier[outlier_indices] = True
        current_frame.egoVx = self.ego_kf_state['x'][0, 0]
        current_frame.egoVy = self.ego_kf_state['x'][1, 0]

        all_indices = np.arange(num_points)
        static_inlier_indices = np.setdiff1d(all_indices, outlier_indices, assume_unique=True)
        is_vehicle_moving = np.abs(current_frame.egoVx) > self.params['ego_motion_params']['stationarySpeedThreshold']
        
        static_box = self.params['stationary_cluster_box']
        
        dynamic_box = None
        if static_inlier_indices.size > 0 and is_vehicle_moving and motion_state == 0:
            dynamic_x_range, self.filtered_barrier_x = detect_side_barrier(
                cartesian_pos_data, static_inlier_indices,
                self.params['barrier_detect_params'], self.filtered_barrier_x
            )
            dynamic_box = {
                'X_RANGE': dynamic_x_range,
                'Y_RANGE': self.params['barrier_detect_params']['longitudinal_range']
            }
        
        current_frame.filtered_barrier_x = type('barrier_struct', (object,), self.filtered_barrier_x)()

        # --- Clustering and Filtering ---
        detected_centroids, detected_cluster_info = np.empty((0, 2)), []
        if num_points >= self.params['dbscan_params']['min_pts']:
            grid_map, point_to_grid_idx = slot_points_to_grid(cartesian_pos_data, self.params['grid_config'])
            current_frame.grid_map = grid_map
            
            dbscan_clusters = my_dbscan(cartesian_pos_data.T, point_cloud[3, :],
                                         self.params['dbscan_params']['epsilon_pos'], 
                                         self.params['dbscan_params']['epsilon_vel'], 
                                         self.params['dbscan_params']['min_pts'], 
                                         grid_map, point_to_grid_idx)
            current_frame.dbscanClusters = dbscan_clusters
            unique_ids = np.unique(dbscan_clusters)
            unique_ids = unique_ids[unique_ids > 0]

            if unique_ids.size > 0:
                temp_centroids, temp_cluster_info = [], []
                stationary_speed_threshold = self.params['ego_motion_params']['stationarySpeedThreshold']

                for cid in unique_ids:
                    cluster_indices = np.where(dbscan_clusters == cid)[0]
                    centroid = np.mean(cartesian_pos_data[:, cluster_indices], axis=1)
                    mean_radial_speed = np.mean(point_cloud[3, cluster_indices])
                    
                    # --- THIS IS THE FIX ---
                    # Correctly determine if a cluster is moving based on the ego vehicle's state.
                    if is_vehicle_moving:
                        # When moving, use RANSAC outliers. A high ratio of outliers means the cluster is moving relative to the static environment.
                        outlier_ratio = np.sum(current_frame.isOutlier[cluster_indices]) / len(cluster_indices)
                        is_moving_cluster = outlier_ratio > self.params['cluster_filter_params']['min_outlierClusterRatio_thrs']
                    else:
                        # When stationary, the world is the reference. Use the cluster's absolute radial speed.
                        is_moving_cluster = abs(mean_radial_speed) > stationary_speed_threshold
                    # --- END OF FIX ---

                    # Filter 1: Dynamic Barriers (Guardrails). Only active when vehicle is moving straight.
                    if dynamic_box is not None:
                        is_in_dynamic_box = (dynamic_box['X_RANGE'][0] <= centroid[0] <= dynamic_box['X_RANGE'][1]) and \
                                            (dynamic_box['Y_RANGE'][0] <= centroid[1] <= dynamic_box['Y_RANGE'][1])
                        if not is_moving_cluster and is_in_dynamic_box:
                            continue 
                    
                    # Filter 2: Static Box (for tracking stopped cars).
                    is_in_static_box = (static_box['X_RANGE'][0] <= centroid[0] <= static_box['X_RANGE'][1]) and \
                                       (static_box['Y_RANGE'][0] <= centroid[1] <= static_box['Y_RANGE'][1])
                    
                    is_stationary_in_box_flag = (not is_moving_cluster) and is_in_static_box
                    
                    azimuth_rad = np.arctan2(centroid[0], centroid[1])
                    temp_centroids.append(centroid)
                    temp_cluster_info.append({
                        'X': centroid[0], 'Y': centroid[1], 'radialSpeed': mean_radial_speed,
                        'vx': mean_radial_speed * np.sin(azimuth_rad), 
                        'vy': mean_radial_speed * np.cos(azimuth_rad),
                        'isOutlierCluster': is_moving_cluster, # This flag now correctly represents motion
                        'isStationary_inBox': is_stationary_in_box_flag
                    })

                if temp_centroids:
                    detected_centroids = np.array(temp_centroids)
                    detected_cluster_info = temp_cluster_info

        current_frame.detectedClusterInfo = np.array(detected_cluster_info, dtype=object) if detected_cluster_info else np.array([])
        
        # --- Perform Tracking ---
        self.all_tracks, self.next_track_id, _, _ = perform_track_assignment_master(
            self.frame_idx, detected_centroids, detected_cluster_info,
            self.all_tracks, self.next_track_id, delta_t, imu_omega, self.params
        )

        self.frame_idx += 1
        return self.all_tracks, current_frame