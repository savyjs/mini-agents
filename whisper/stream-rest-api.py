import whisper
import subprocess
import os
import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional
import tempfile
import asyncio
import sys

import uvicorn
from fastapi import FastAPI
import socketio

# -----------------------
# Logging Configuration
# -----------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# -----------------------
# FastAPI and Socket.IO Initialization
# -----------------------
app = FastAPI(title="Video Transcription API")
sio = socketio.AsyncServer(cors_allowed_origins=["http://localhost:3000"], async_mode='asgi')
app = socketio.ASGIApp(sio, app)

# -----------------------
# Whisper Model Load
# -----------------------
try:
    model = whisper.load_model("turbo")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    sys.exit(1)

# -----------------------
# Global State for Current Session
# -----------------------
current_session: Optional[str] = None
temp_dir: Optional[Path] = None
audio_buffer: bytes = b''  # Buffer to accumulate streaming data
last_process_time: float = 0.0
PROCESS_INTERVAL = 2.0  # Process buffer every 2 seconds
header_received = False  # Track if header chunk is received

# -----------------------
# Utility Functions
# -----------------------
def setup_temp_dir(session_id: str) -> Path:
    global temp_dir
    temp_dir = Path(tempfile.mkdtemp(prefix=f"session_{session_id}_"))
    return temp_dir

def cleanup_temp_dir():
    global temp_dir, audio_buffer, header_received
    if temp_dir and temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.debug(f"Cleaned up temp directory: {temp_dir}")
    audio_buffer = b''
    temp_dir = None
    header_received = False

async def convert_and_transcribe_buffer() -> None:
    global audio_buffer, last_process_time, header_received
    current_time = asyncio.get_event_loop().time()
    if current_time - last_process_time < PROCESS_INTERVAL or not audio_buffer or not header_received:
        return

    chunk_path = None
    wav_path = None
    try:
        chunk_path = temp_dir / f"stream_chunk_{uuid.uuid4().hex}.webm"
        with open(chunk_path, "wb") as f:
            f.write(audio_buffer)
            audio_buffer = b''  # Clear buffer after writing

        wav_path = temp_dir / "stream_chunk.wav"
        ffmpeg_command = [
            "ffmpeg", "-y", "-f", "webm", "-i", str(chunk_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(wav_path)
        ]
        logger.debug(f"Running ffmpeg: {' '.join(ffmpeg_command)}")
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise Exception(f"FFmpeg conversion failed: {stderr.decode()}")

        logger.debug(f"Transcribing buffer from {wav_path}")
        result = model.transcribe(str(wav_path), word_timestamps=True)
        for segment in result["segments"]:
            await sio.emit('transcription_segment', {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            }, room=current_session)
    except Exception as e:
        logger.error(f"Error processing stream buffer: {e}")
        await sio.emit('error', f"Error processing stream: {str(e)}", room=current_session)
    finally:
        if chunk_path and chunk_path.exists():
            os.unlink(chunk_path)
        if wav_path and wav_path.exists():
            os.unlink(wav_path)
        last_process_time = current_time

# -----------------------
# Socket.IO Event Handlers
# -----------------------
@sio.event
async def connect(sid, environ):
    logger.debug(f"Client connected: {sid}")
    global current_session
    current_session = sid
    setup_temp_dir(sid)

@sio.event
async def disconnect(sid):
    logger.debug(f"Client disconnected: {sid}")
    if current_session == sid:
        cleanup_temp_dir()

@sio.event
async def video_stream(sid, data):
    global audio_buffer, header_received
    if current_session != sid:
        logger.warning(f"Unauthorized stream from {sid}")
        await sio.emit('error', "Unauthorized session.", room=sid)
        return

    if isinstance(data, dict):
        if data.get('isHeader'):
            audio_buffer = bytes(data['data'])  # Convert list to bytes for header chunk
            header_received = True
            logger.debug("Received header chunk")
        elif header_received:
            audio_buffer += bytes(data['data'])  # Convert list to bytes for subsequent data
            logger.debug(f"Appended data, buffer size: {len(audio_buffer)}")
        else:
            logger.warning("Header not received before data")
            await sio.emit('error', "Header not received before data.", room=sid)
    else:
        logger.warning("Invalid data format")
        await sio.emit('error', "Invalid stream data format.", room=sid)
        return

    await convert_and_transcribe_buffer()  # Process buffer periodically

@sio.event
async def end_recording(sid):
    if current_session == sid:
        await convert_and_transcribe_buffer()  # Process any remaining data
        cleanup_temp_dir()
        await sio.emit('end_recording', "Recording ended.", room=sid)

# -----------------------
# Uvicorn Launch
# -----------------------
if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=9000)
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)
