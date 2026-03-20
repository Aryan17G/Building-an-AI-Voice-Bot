# training/evaluate_metrics.py
import whisper
from jiwer import wer
import os

def evaluate_asr():
    print("Loading Whisper 'base' model for evaluation...")
    model = whisper.load_model("base")

    # 1. Define your evaluation dataset
    # You will need to record a few short WAV files and put them in a folder named 'test_audio'
    # Update the "truth" strings to match exactly what you said in the recordings.
    eval_data = [
        # Change .wav to .m4a if that is what your files are!
        {"file": "test_audio/order_status.m4a", "truth": "where is my package"},
        {"file": "test_audio/refund.m4a", "truth": "i need a refund please"},
        {"file": "test_audio/noisy_cancel.m4a", "truth": "can you cancel my subscription"}
    ]

    predictions = []
    references = []

    print("\nStarting transcription...")
    for item in eval_data:
        if not os.path.exists(item["file"]):
            print(f"Warning: File {item['file']} not found. Skipping.")
            continue
            
        # Transcribe the audio file
        result = model.transcribe(item["file"])
        
        # Clean up text (lowercase and strip whitespace for accurate comparison)
        pred_text = result["text"].strip().lower()
        truth_text = item["truth"].lower()
        
        predictions.append(pred_text)
        references.append(truth_text)
        
        print(f"File: {item['file']}")
        print(f"  Truth:     {truth_text}")
        print(f"  Predicted: {pred_text}\n")

    # 2. Calculate WER
    if predictions and references:
        error_rate = wer(references, predictions)
        print(f"=====================================")
        print(f"Final Word Error Rate (WER): {error_rate:.2%}")
        print(f"=====================================")
    else:
        print("No audio files were processed. Please check your 'test_audio' folder.")

if __name__ == "__main__":
    evaluate_asr()