from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import logging
import subprocess
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify your frontend URL)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Whisper model and processor
model_name = "walishajeeh/whisper-small-en"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Enable GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def convert_webm_to_wav(webm_path, wav_path):
    try:
        command = f"ffmpeg -i {webm_path} -ar 16000 -ac 1 -y {wav_path}"
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion error: {e}")
        raise HTTPException(status_code=500, detail="Audio conversion failed.")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}")

        # Save the uploaded file temporarily
        webm_path = "temp.webm"
        wav_path = "temp.wav"

        file_content = await file.read()
        with open(webm_path, "wb") as f:
            f.write(file_content)

        # Convert webm to wav
        convert_webm_to_wav(webm_path, wav_path)

        # Read and resample the audio file to 16kHz
        audio_input, sample_rate = librosa.load(wav_path, sr=16000)
        audio_input = audio_input.astype(np.float32)  # Ensure correct dtype

        input_features = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)

        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Clean up temporary files
        os.remove(webm_path)
        os.remove(wav_path)

        return {"transcription": transcription}
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
