from transformers import pipeline
from app.logger import logger
from app.config import settings

class NLUService:
    def __init__(self):
        logger.info(f"Loading NLU Model from {settings.INTENT_MODEL_PATH}...")
        self.classifier = pipeline(
            "text-classification", 
            model=settings.INTENT_MODEL_PATH, 
            tokenizer=settings.INTENT_MODEL_PATH, 
            return_all_scores=False
        )

    def predict_intent(self, text: str):
        try:
            logger.info(f"Predicting intent for: '{text}'")
            result = self.classifier(text)[0]
            return {
                "intent": result["label"],
                "confidence": result["score"]
            }
        except Exception as e:
            logger.error(f"NLU Error: {e}")
            raise