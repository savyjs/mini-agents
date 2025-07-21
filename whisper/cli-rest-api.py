import whisper
import subprocess
import os
import argparse
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="Video Transcription API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic model for response structure
class TranscriptionResponse(BaseModel):
    text: str
    segments: list[dict]


def transcribe_video(mp4_path: str) -> Dict[str, Any]:
    """Core transcription function that processes an MP4 file."""
    try:
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav_path = temp_wav.name

        # Extract audio with FFmpeg
        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-i", mp4_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            wav_path
        ]
        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)

        # Transcribe using Whisper
        model = whisper.load_model("small.en")
        result = model.transcribe(wav_path, word_timestamps=True)

        # Clean up temporary file
        os.unlink(wav_path)

        return {
            "text": result["text"],
            "segments": [
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"]
                }
                for segment in result["segments"]
            ]
        }

    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg error: {e.stderr}")
    except Exception as e:
        raise Exception(f"Transcription error: {str(e)}")


# REST API Endpoint
@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_endpoint(file: UploadFile = File(...)):
    """REST API endpoint to transcribe uploaded MP4 file."""
    # Validate file type by checking content type and extension
    if not file.content_type.startswith("video/mp4") and not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 files are supported")

    try:
        # Create temporary directory for file handling
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_mp4_path = os.path.join(temp_dir, file.filename or "uploaded_video.mp4")

            # Write uploaded file to temporary path
            with open(temp_mp4_path, "wb") as temp_file:
                shutil.copyfileobj(file.file, temp_file)

            # Process transcription
            result = transcribe_video(temp_mp4_path)

        return TranscriptionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()


def cli_main():
    """Command-line interface for video transcription."""
    parser = argparse.ArgumentParser(description="Transcribe audio from MP4 video files")
    parser.add_argument("file", help="Path to the MP4 file")
    args = parser.parse_args()

    if not args.file.lower().endswith(".mp4"):
        print("❌ Error: Input file must be an MP4 file")
        return

    if not os.path.exists(args.file):
        print(f"❌ Error: File {args.file} does not exist")
        return

    try:
        print(f"📂 Processing file: {args.file}")
        result = transcribe_video(args.file)

        print("\n📜 Final Transcription:")
        print(result["text"])
        print("\n🕓 Segments with Timestamps:")
        for segment in result["segments"]:
            print(f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    import sys

    # Check if running in CLI mode
    if len(sys.argv) > 1:
        cli_main()
    else:
        # Run FastAPI server
        uvicorn.run(app, host="0.0.0.0", port=8000)
