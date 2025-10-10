# main.py

import sys
from src.console_logger import setup_logging

def main():
    """
    Main entry point for the Unified Radar Tracker application.
    Allows the user to select between live tracking and playback mode.
    """
    print("--- Welcome to the Unified Radar Tracker ---")
    
    while True:
        mode = input("Select mode: (1) Live Tracking or (2) Playback from File\nEnter choice (1 or 2): ")
        if mode in ['1', '2']:
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    # --- Setup shared logging for both modes ---
    setup_logging()

    if mode == '1':
        print("\nStarting in LIVE mode...")
        # We import here to avoid PyQt5 dependency if only running playback
        from src.main_live import main as main_live
        main_live()
    elif mode == '2':
        print("\nStarting in PLAYBACK mode...")
        from src.main_playback import run_playback
        run_playback()

if __name__ == '__main__':
    # Add src to path to allow for relative imports
    sys.path.append('src')
    main()