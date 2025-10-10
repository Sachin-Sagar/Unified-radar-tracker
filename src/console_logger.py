# src/console_logger.py

import logging
import json
from datetime import datetime

class JsonFormatter(logging.Formatter):
    """
    Formats log records as JSON objects.
    """
    def format(self, record):
        log_object = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineNo": record.lineno
        }
        return json.dumps(log_object)

def setup_logging():
    """
    Configures logging to output to both the console and two separate files 
    (one plain text, one JSON).
    """
    log_filename_txt = f"output/console_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_filename_json = f"output/console_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # --- Create Handlers ---
    # 1. Console Handler (for live viewing)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    # 2. Text File Handler (for easy reading)
    file_handler_txt = logging.FileHandler(log_filename_txt, mode='w')
    file_handler_txt.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 3. JSON File Handler (for post-processing)
    file_handler_json = logging.FileHandler(log_filename_json, mode='w')
    file_handler_json.setFormatter(JsonFormatter())

    # --- Configure Root Logger ---
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            console_handler,
            file_handler_txt,
            file_handler_json
        ]
    )
    logging.info("Logging configured for console, text file, and JSON file.")