import whisper
import subprocess
import os
import uuid
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile

# -----------------------
# Logging Configuration
# -----------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# -----------------------
# FastAPI Initialization
# -----------------------
app = FastAPI(title="Video Transcription API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# MIME Type Check
# -----------------------
ALLOWED_MIME_TYPES = {"video/mp4", "audio/mpeg", "video/webm"}
ALLOWED_EXTENSIONS = {".mp4", ".mp3", ".webm"}

def is_valid_file(file: UploadFile) -> bool:
    ext = Path(file.filename).suffix.lower()
    return file.content_type in ALLOWED_MIME_TYPES and ext in ALLOWED_EXTENSIONS

# -----------------------
# Whisper Model Load
# -----------------------
model = whisper.load_model("small.en")

# -----------------------
# Response Model
# -----------------------
class TranscriptionResponse(BaseModel):
    text: str
    segments: list[dict]

# -----------------------
# Core Transcription Logic
# -----------------------
def transcribe_media(media_path: str) -> Dict[str, Any]:
    ext = Path(media_path).suffix.lower()

    if ext == ".mp3":
        audio_path = media_path
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            audio_path = temp_wav.name

        ffmpeg_command = [
            "ffmpeg", "-y", "-i", media_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_path
        ]
        logger.debug(f"Running ffmpeg: {' '.join(ffmpeg_command)}")
        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)

    try:
        logger.debug(f"Starting transcription on {audio_path}")
        result = model.transcribe(audio_path, word_timestamps=True)
    finally:
        if ext != ".mp3" and os.path.exists(audio_path):
            os.unlink(audio_path)
            logger.debug(f"Deleted temporary WAV file: {audio_path}")

    return {
        "text": result["text"],
        "segments": [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in result["segments"]
        ]
    }

# -----------------------
# Upload Endpoint
# -----------------------
@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    if not is_valid_file(file):
        raise HTTPException(status_code=400, detail="Only MP4, MP3, and WEBM files are supported.")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_id = uuid.uuid4().hex[:8]
    original_ext = Path(file.filename).suffix.lower()
    filename = f"input_{timestamp}_{random_id}{original_ext}"

    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / filename

    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            f.flush()
            os.fsync(f.fileno())

        logger.debug(f"Saved uploaded file: {temp_path} ({os.stat(temp_path).st_size} bytes)")

        if not temp_path.exists() or os.stat(temp_path).st_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty or corrupted.")

        result = transcribe_media(str(temp_path))
        return TranscriptionResponse(**result)

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr}")
        raise HTTPException(status_code=500, detail="Error converting file with FFmpeg.")

    except Exception as e:
        logger.exception("Unexpected error during transcription.")
        raise HTTPException(status_code=500, detail="Internal server error.")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        await file.close()

# -----------------------
# Uvicorn Launch
# -----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
