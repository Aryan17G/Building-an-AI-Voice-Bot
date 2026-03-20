from gtts import gTTS
import io
from app.logger import logger

class TTSService:
    def synthesize(self, text: str) -> io.BytesIO:
        try:
            logger.info(f"Synthesizing speech for: '{text}'")
            tts = gTTS(text=text, lang='en', slow=False)
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            return audio_fp
        except Exception as e:
            logger.error(f"TTS Error: {e}")
            raise