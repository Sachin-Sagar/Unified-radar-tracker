# src/main_live.py

import sys
import time
from datetime import datetime
import numpy as np
import logging  # <-- Add logging import

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal, QObject

# --- Import project modules ---
from .hardware import hw_comms_utils, parsing_utils
from .hardware.read_and_parse_frame import read_and_parse_frame
from .data_adapter import adapt_frame_data_to_fhist
from .tracking.tracker import RadarTracker
from .tracking.parameters import define_parameters
from .tracking.update_and_save_history import update_and_save_history
from .live_visualizer import LiveVisualizer
from .json_logger import DataLogger
from .console_logger import setup_logging  # <-- Import the new setup function

# --- Configuration ---
if sys.platform == "win32":
    CLI_COMPORT_NUM = 'COM11'
elif sys.platform == "linux":
    CLI_COMPORT_NUM = '/dev/ttyACM0'
else:
    logging.error(f"Unsupported OS '{sys.platform}' detected. Please set COM port manually.") # <-- Use logging
    CLI_COMPORT_NUM = None
    
CONFIG_FILE_PATH = 'configs/profile_80_m_40mpsec_bsdevm_16tracks_dyClutter.cfg'
INITIAL_BAUD_RATE = 115200

class RadarWorker(QObject):
    """
    This worker runs the main radar processing loop and also manages
    the JSON data logger.
    """
    frame_ready = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(self, cli_com_port, config_file):
        super().__init__()
        self.cli_com_port = cli_com_port
        self.config_file = config_file
        self.is_running = True
        self.params_radar = None
        self.h_data_port = None
        self.tracker = None
        self.fhist_history = []
        self.logger_thread = None
        self.data_logger = None

    def run(self):
        """The main processing loop."""
        log_filename = f"output/radar_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.logger_thread = QThread()
        self.data_logger = DataLogger(log_filename)
        self.data_logger.moveToThread(self.logger_thread)
        self.logger_thread.started.connect(self.data_logger.run)
        self.logger_thread.start()

        self.params_radar, self.h_data_port = self._configure_sensor()
        if not self.params_radar or not self.h_data_port:
            logging.error("Failed to configure sensor. Exiting worker thread.") # <-- Use logging
            self.stop()
            self.finished.emit()
            return

        params_tracker = define_parameters()
        self.tracker = RadarTracker(params_tracker)

        logging.info("--- Starting Live Tracking ---") # <-- Use logging
        while self.is_running:
            frame_data = read_and_parse_frame(self.h_data_port, self.params_radar)
            if not frame_data or not frame_data.header:
                continue
            
            self.data_logger.add_data(frame_data)

            fhist_frame = adapt_frame_data_to_fhist(frame_data, self.tracker.last_timestamp_ms)
            updated_tracks, processed_frame = self.tracker.process_frame(fhist_frame, can_signals=None)
            self.fhist_history.append(processed_frame)

            num_confirmed_tracks = sum(1 for t in updated_tracks if t.get('isConfirmed') and not t.get('isLost'))
            logging.info(f"Frame: {self.tracker.frame_idx} | Detections: {frame_data.num_points} | Confirmed Tracks: {num_confirmed_tracks}") # <-- Use logging

            self.frame_ready.emit(frame_data)

        self._save_tracking_history()
        self.finished.emit()

    def _configure_sensor(self):
        """Reads the config file and sends commands to the radar."""
        cli_cfg = parsing_utils.read_cfg(self.config_file)
        if not cli_cfg: return None, None
        params = parsing_utils.parse_cfg(cli_cfg)
        target_baud_rate = INITIAL_BAUD_RATE
        for command in cli_cfg:
            if command.startswith("baudRate"):
                try: target_baud_rate = int(command.split()[1])
                except (ValueError, IndexError): pass
                break
        logging.info("\n--- Starting Sensor Configuration ---") # <-- Use logging
        h_port = hw_comms_utils.configure_control_port(self.cli_com_port, INITIAL_BAUD_RATE)
        if not h_port: return None, None
        for command in cli_cfg:
            logging.info(f"> {command}") # <-- Use logging
            h_port.write((command + '\n').encode())
            time.sleep(0.1)
            if "baudRate" in command:
                time.sleep(0.2)
                try:
                    h_port.baudrate = target_baud_rate
                    logging.info(f"  Baud rate changed to {target_baud_rate}") # <-- Use logging
                except Exception as e:
                    logging.error(f"ERROR: Failed to change baud rate: {e}") # <-- Use logging
                    h_port.close()
                    return None, None
        logging.info("--- Configuration complete ---\n") # <-- Use logging
        hw_comms_utils.reconfigure_port_for_data(h_port)
        return params, h_port

    def _save_tracking_history(self):
        """Saves the final processed tracking history."""
        logging.info("\n--- Saving tracking history ---") # <-- Use logging
        if self.tracker and self.fhist_history:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output/track_history_{timestamp}.json"
            update_and_save_history(
                self.tracker.all_tracks,
                self.fhist_history,
                filename,
                params=self.tracker.params
            )
        else:
            logging.warning("No frame history was processed, nothing to save.") # <-- Use logging

    def stop(self):
        """Stops the processing loop and the logger."""
        logging.info("--- Stopping worker thread... ---") # <-- Use logging
        self.is_running = False
        
        if self.data_logger:
            self.data_logger.stop()
        if self.logger_thread:
            self.logger_thread.quit()
            self.logger_thread.wait()

        if self.h_data_port and self.h_data_port.is_open:
            self.h_data_port.close()
            logging.info("--- Serial port closed ---") # <-- Use logging

def main():
    """Main application entry point."""
    setup_logging()  # <-- Initialize the logger here
    app = QApplication(sys.argv)
    worker_thread = QThread()
    radar_worker = RadarWorker(CLI_COMPORT_NUM, CONFIG_FILE_PATH)
    radar_worker.moveToThread(worker_thread)
    visualizer = LiveVisualizer(radar_worker, worker_thread)
    visualizer.show()
    worker_thread.started.connect(radar_worker.run)
    radar_worker.frame_ready.connect(visualizer.update_plot)
    radar_worker.finished.connect(worker_thread.quit)
    radar_worker.finished.connect(radar_worker.deleteLater)
    worker_thread.finished.connect(worker_thread.deleteLater)
    worker_thread.start()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()