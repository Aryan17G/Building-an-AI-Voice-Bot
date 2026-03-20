from app.logger import logger
from app.config import settings

class NLGService:
    def __init__(self):
        self.responses = {
            "order_status": "Your order is currently being processed and will ship within two business days.",
            "refund_request": "I can help you with a refund. Please ensure you have your order number ready.",
            "cancel_order": "To cancel your order, I will need to transfer you to a human agent. Please hold.",
            "speak_to_agent": "Please hold while I connect you to the next available agent.",
            "fallback": "I'm sorry, I didn't quite catch that. Could you please repeat your request?"
        }

    def generate_response(self, intent: str, confidence: float) -> str:
        logger.info(f"Generating response for intent: {intent} (Confidence: {confidence:.2f})")
        
        if confidence < settings.CONFIDENCE_THRESHOLD:
            logger.warning("Confidence below threshold. Using fallback response.")
            return self.responses["fallback"]
            
        return self.responses.get(intent, self.responses["fallback"])