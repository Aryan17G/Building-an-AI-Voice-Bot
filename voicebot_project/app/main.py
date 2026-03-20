from fastapi import FastAPI, UploadFile, File, HTTPException , Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import shutil
import os
import time

# Import configurations, logger, and services
from app.logger import logger
from app.services.asr_service import ASRService
from app.services.nlu_service import NLUService
from app.services.nlg_service import NLGService
from app.services.tts_service import TTSService

app = FastAPI(title="Customer Support Voice Bot API")

# Initialize services globally
logger.info("Initializing application services...")
asr = ASRService()
nlu = NLUService()
nlg = NLGService()
tts = TTSService()
logger.info("All services initialized successfully.")

# Pydantic models for text endpoints
class TextRequest(BaseModel):
    text: str

class IntentRequest(BaseModel):
    intent: str
    confidence: float

# --- INDIVIDUAL ENDPOINTS ---

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    temp_file = f"temp_{audio.filename}"
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        text = asr.transcribe(temp_file)
        return {"text": text}
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

@app.post("/predict-intent")
async def predict_intent(request: TextRequest):
    return nlu.predict_intent(request.text)

@app.post("/generate-response")
async def generate_response(request: IntentRequest):
    response_text = nlg.generate_response(request.intent, request.confidence)
    return {"response": response_text}

@app.post("/synthesize")
async def synthesize_text(request: TextRequest):
    audio_stream = tts.synthesize(request.text)
    return StreamingResponse(audio_stream, media_type="audio/mpeg")

# --- UNIFIED END-TO-END ENDPOINT ---

# 1. Global variable to hold the latest audio in memory
latest_audio_bytes = b""

@app.post("/voicebot")
async def process_voice_request(audio: UploadFile = File(...)):
    global latest_audio_bytes
    start_time = time.time()
    temp_file = f"temp_{audio.filename}"
    logger.info(f"Received voicebot request. Processing file: {audio.filename}")
    
    try:
        # Save File
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
            
        # Pipeline
        text = asr.transcribe(temp_file)
        intent_data = nlu.predict_intent(text)
        response_text = nlg.generate_response(intent_data["intent"], intent_data["confidence"])
        audio_stream = tts.synthesize(response_text)
        
        # Save the audio stream to our global variable
        latest_audio_bytes = audio_stream.read()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Successfully processed request in {elapsed_time:.2f} seconds.")
        
        # Return a standard Response
        return Response(content=latest_audio_bytes, media_type="audio/mpeg")
        
    except Exception as e:
        logger.error(f"Failed to process request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

# 2. Add the GET endpoint that the browser's audio player is secretly asking for!
@app.get("/voicebot")
async def play_latest_voice():
    global latest_audio_bytes
    return Response(content=latest_audio_bytes, media_type="audio/mpeg")