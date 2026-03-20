import whisper
import torch
from app.logger import logger
from app.config import settings

class ASRService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper '{settings.WHISPER_MODEL_SIZE}' on {self.device}...")
        self.model = whisper.load_model(settings.WHISPER_MODEL_SIZE, device=self.device)

    def transcribe(self, audio_path: str) -> str:
        try:
            logger.info(f"Transcribing audio file...")
            result = self.model.transcribe(audio_path)
            return result["text"].strip()
        except Exception as e:
            logger.error(f"ASR Error: {e}")
            raise