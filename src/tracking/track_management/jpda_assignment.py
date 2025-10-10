# src/track_management/jpda_assignment.py

import numpy as np
import logging

from ..algorithms.find_jpda_hypotheses import find_jpda_hypotheses
from ..filters.imm_filter import imm_correct

def jpda_assignment(
    all_tracks, active_track_indices, detected_centroids, detected_cluster_info,
    kf_measurement_noise, PD, lambda_c, gating_chi2, position_gating_threshold,
    params
):
    """
    Performs JPDA for confirmed tracks, including gating, hypothesis generation,
    and a full probabilistic IMM state update.
    """
    debug_mode = params.get('debug_mode', False)
    debug_mode1 = params.get('debug_mode1', False)
    
    num_detections = detected_centroids.shape[0] if detected_centroids is not None else 0
    num_active_tracks = len(active_track_indices)
    
    if num_active_tracks == 0:
        return all_tracks, np.array([]), np.array([]), np.array([])

    miss_flags = np.ones(num_active_tracks, dtype=bool)

    if debug_mode:
        logging.info(f'[JPDA] Step 1: Performing Gating for {num_active_tracks} tracks and {num_detections} measurements.')

    # --- Step 1: Gating ---
    validation_matrix = np.zeros((num_detections, num_active_tracks), dtype=bool)
    likelihoods = np.zeros((num_detections, num_active_tracks))
    
    for t, track_idx in enumerate(active_track_indices):
        track = all_tracks[track_idx]
        x_pred_comb = track['immState']['x']
        P_pred_comb = track['immState']['P']
        pred_x, pred_y = x_pred_comb[0, 0], x_pred_comb[1, 0]

        if debug_mode:
            logging.info(f'\n[JPDA-GATING-DEBUG] Gating for Track ID {track["id"]} at predicted pos ({pred_x:.1f}, {pred_y:.1f}):')

        for d in range(num_detections):
            det_x, det_y = detected_centroids[d, 0], detected_centroids[d, 1]
            
            euc_dist = np.sqrt((pred_x - det_x)**2 + (pred_y - det_y)**2)
            if euc_dist > position_gating_threshold:
                if debug_mode1:
                    logging.info(f'  [GATING-FAIL] Track {track["id"]} vs Meas {d}: Failed Euclidean gate. Dist: {euc_dist:.2f} > Thresh: {position_gating_threshold:.2f}')
                continue

            z = np.array([np.sqrt(det_x**2 + det_y**2), np.arctan2(det_x, det_y), 
                          detected_cluster_info[d]['radialSpeed']]).reshape(3, 1)

            r_pred = np.sqrt(pred_x**2 + pred_y**2)
            if r_pred < 1e-6: continue

            h_x = np.array([r_pred, np.arctan2(pred_x, pred_y), 
                            (pred_x * x_pred_comb[2,0] + pred_y * x_pred_comb[3,0]) / r_pred]).reshape(3,1)
            
            y = z - h_x
            y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi

            px, py, vx, vy = pred_x, pred_y, x_pred_comb[2,0], x_pred_comb[3,0]
            r, r_sq, r_cub = r_pred, r_pred**2, r_pred**3
            H = np.zeros((3, 7))
            H[0, 0] = px/r; H[0, 1] = py/r
            H[1, 0] = py/r_sq; H[1, 1] = -px/r_sq
            H[2, 0] = (vx/r)-(px*(px*vx+py*vy))/r_cub; H[2, 1] = (vy/r)-(py*(px*vx+py*vy))/r_cub
            H[2, 2] = px/r; H[2, 3] = py/r

            P_gate = P_pred_comb[:4, :4]
            H_gate = H[:, :4]
            S = H_gate @ P_gate @ H_gate.T + kf_measurement_noise
            
            try:
                S_inv = np.linalg.inv(S)
                mahalanobis_dist_sq = y.T @ S_inv @ y
            except np.linalg.LinAlgError:
                continue
            
            if mahalanobis_dist_sq <= gating_chi2:
                validation_matrix[d, t] = True
                likelihoods[d, t] = np.exp(-0.5 * mahalanobis_dist_sq) / np.sqrt(np.linalg.det(2 * np.pi * S))
            elif debug_mode1:
                logging.info(f'  [GATING-FAIL] Track {track["id"]} vs Meas {d}: Failed Mahalanobis gate. Dist^2: {mahalanobis_dist_sq[0,0]:.2f} > Thresh: {gating_chi2:.2f}')

    if debug_mode:
        logging.info('[JPDA-DEBUG] Gating complete. Validation Matrix (Rows: Meas, Cols: Tracks):\n' + str(validation_matrix.astype(int)))
    
    if debug_mode: logging.info('[JPDA] Step 2: Generating joint association hypotheses.')
    hypotheses = find_jpda_hypotheses(validation_matrix, params)
    if debug_mode: logging.info(f'[JPDA] -> Found {len(hypotheses)} possible hypotheses.')

    if not hypotheses:
        return all_tracks, miss_flags, validation_matrix, np.zeros((num_detections + 1, num_active_tracks))

    if debug_mode:
        logging.info('[JPDA] Step 3: Calculating hypothesis probabilities.')

    hypothesis_probs = np.zeros(len(hypotheses))
    for h_idx, hypo in enumerate(hypotheses):
        prob = 1.0
        for t_idx, meas_idx in enumerate(hypo):
            if meas_idx > 0:
                prob *= PD * likelihoods[meas_idx - 1, t_idx]
            else:
                prob *= (1 - PD)
        
        unassigned_meas = set(range(1, num_detections + 1)) - set(m for m in hypo if m > 0)
        prob *= (lambda_c ** len(unassigned_meas))
        hypothesis_probs[h_idx] = prob
        
    total_prob = np.sum(hypothesis_probs)
    hypothesis_probs = hypothesis_probs / total_prob if total_prob > 0 else np.ones(len(hypotheses)) / len(hypotheses)
    
    if debug_mode: logging.info('[JPDA] Step 4: Calculating marginal probabilities (beta).')
    beta = np.zeros((num_detections + 1, num_active_tracks))
    for h_idx, hypo in enumerate(hypotheses):
        for t_idx, meas_idx in enumerate(hypo):
            beta[meas_idx, t_idx] += hypothesis_probs[h_idx]

    if debug_mode:
        logging.info('[JPDA-DEBUG] Final Beta Matrix (Rows: M0, M1...; Cols: T1, T2...):\n' + np.array2string(beta, formatter={'float_kind':lambda x: "%.4f" % x}))

    if debug_mode: logging.info('[JPDA] Step 5: Performing probabilistic IMM state update.')
    
    for t_idx, track_idx in enumerate(active_track_indices):
        track = all_tracks[track_idx]
        imm_state_pred = track['immState']
        beta_i0 = beta[0, t_idx]

        hypo_states = {0: imm_state_pred}
        
        for d in range(num_detections):
            if validation_matrix[d, t_idx] and beta[d + 1, t_idx] > 1e-9:
                det_x, det_y = detected_centroids[d, 0], detected_centroids[d, 1]
                z_polar = np.array([np.sqrt(det_x**2+det_y**2), np.arctan2(det_x, det_y), 
                                    detected_cluster_info[d]['radialSpeed']]).reshape(3, 1)
                hypo_states[d + 1] = imm_correct(imm_state_pred, z_polar, kf_measurement_noise)

        x_final = sum(beta[d, t_idx] * state['x'] for d, state in hypo_states.items() if state is not None)
        mu_final = sum(beta[d, t_idx] * state['modelProbabilities'] for d, state in hypo_states.items() if state is not None)
        
        P_final = np.zeros_like(imm_state_pred['P'])
        for d, state in hypo_states.items():
            if state is not None:
                diff = state['x'] - x_final
                P_final += beta[d, t_idx] * (state['P'] + diff @ diff.T)

        all_tracks[track_idx]['immState']['x'] = x_final
        all_tracks[track_idx]['immState']['P'] = P_final
        all_tracks[track_idx]['immState']['modelProbabilities'] = mu_final

        for m in range(3):
            x_m_final = sum(beta[d, t_idx] * state['models'][m]['x'] for d, state in hypo_states.items() if state is not None)
            P_m_final = np.zeros_like(imm_state_pred['models'][m]['P'])
            for d, state in hypo_states.items():
                if state is not None:
                    diff_m = state['models'][m]['x'] - x_m_final
                    P_m_final += beta[d, t_idx] * (state['models'][m]['P'] + diff_m @ diff_m.T)

            all_tracks[track_idx]['immState']['models'][m]['x'] = x_m_final
            all_tracks[track_idx]['immState']['models'][m]['P'] = P_m_final
            
        miss_prob_threshold = 0.75
        max_assoc_prob = np.max(beta[1:, t_idx]) if num_detections > 0 else 0
        miss_flags[t_idx] = not (beta_i0 < miss_prob_threshold and max_assoc_prob > 0.1)
        
        if debug_mode:
            logging.info(f'  [JPDA-DEBUG] Track {t_idx} (ID {track["id"]}) Miss Flag Check:')
            logging.info(f'    - Beta_i0 (Miss Prob): {beta_i0:.4f}')
            logging.info(f'    - Max Assoc Prob:      {max_assoc_prob:.4f}')
            logging.info(f'    - Is Miss?             {miss_flags[t_idx]}')
        
    return all_tracks, miss_flags, validation_matrix, beta