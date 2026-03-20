# app/logger.py
import logging
import sys
from app.config import settings

def setup_logger():
    # Create a custom logger
    logger = logging.getLogger("VoiceBotLogger")
    logger.setLevel(settings.LOG_LEVEL)

    # Prevent duplicating logs if this runs multiple times
    if not logger.handlers:
        # 1. Console Handler (prints to terminal)
        c_handler = logging.StreamHandler(sys.stdout)
        
        # 2. File Handler (saves to voicebot.log)
        f_handler = logging.FileHandler(settings.LOG_FILE_PATH)

        # 3. Create a clean format for the logs
        log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        c_handler.setFormatter(log_format)
        f_handler.setFormatter(log_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger

# Export the logger so other files can use it
logger = setup_logger()