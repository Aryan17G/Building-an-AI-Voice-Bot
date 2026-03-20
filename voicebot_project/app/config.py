import os

class Config:
    # Model Configurations
    WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
    
    # Use abspath to force Hugging Face to read this as a local folder!
    INTENT_MODEL_PATH = os.getenv("INTENT_MODEL_PATH", "distilbert-base-uncased")
    
    # Logic Configurations
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.60))
    
    # Application Configurations
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE_PATH = "voicebot.log"

settings = Config()