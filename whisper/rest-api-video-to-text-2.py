import whisper
import subprocess
import os
import uuid
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import eng_to_ipa as ipa
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from numba.core.ir import Print
from pydantic import BaseModel
import tempfile
import torch
import torchaudio
import json
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from huggingface_hub import hf_hub_download
import numpy as np

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

# -------------------------------
# Config
# -------------------------------
PHONEME_MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"
FRAME_STRIDE = 0.02  # seconds per frame
SILENCE_THRESHOLD = 0.2  # Silence duration for word separation
MIN_WORD_DURATION = 0.1  # Minimum duration for a word
MIN_GAP_DURATION = 0.05  # Reduced for better gap detection
ENERGY_THRESHOLD_FACTOR = 2  # Factor for stress detection
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/transformers")

# -------------------------------
# Phoneme → IPA mapping
# -------------------------------
phoneme_to_ipa = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɝ",
    "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
    "AEː": "æː", "AHː": "ɐː", "IYː": "iː", "UWː": "uː", "ERː": "ɚ",
    "OI": "ɔɪ", "OU": "oʊ", "AI": "aɪ", "AU": "aʊ",
    "TS": "ts", "Tʃ": "tʃ", "Dʒ": "dʒ", "ɡʲ": "ɡʲ",
    "NY": "ɲ", "LH": "ɬ", "LL": "ʎ", "RH": "ʁ",
    "SIL": "<s>", "PAD": "<pad>", "EOS": "</s>", "UNK": "<unk>"
}

# Create the inverse mapping
ipa_to_phoneme = {ipa: phoneme for phoneme, ipa in phoneme_to_ipa.items()}

# -------------------------------
# Load phoneme vocab from model
# -------------------------------
print("Loading phoneme vocab...")
vocab_file = hf_hub_download(repo_id=PHONEME_MODEL_NAME, filename="vocab.json", cache_dir=CACHE_DIR)
vocab_path = Path(vocab_file)
if not vocab_path.exists():
    raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

with open(vocab_path, "r", encoding="utf-8") as f:
    phoneme_vocab = json.load(f)

id2phoneme = {v: k for k, v in phoneme_vocab.items()}
print(f"Phoneme vocab loaded: {len(id2phoneme)} phonemes.")
# print("id2phoneme:", id2phoneme)


# -------------------------------
# Load model + processor
# -------------------------------
print("Loading model and processor...")
phoneme_processor = Wav2Vec2Processor.from_pretrained(PHONEME_MODEL_NAME, cache_dir=CACHE_DIR, do_phonemize=False)
phoneme_model = Wav2Vec2ForCTC.from_pretrained(PHONEME_MODEL_NAME, cache_dir=CACHE_DIR)
phoneme_model.eval()
print("Model loaded.\n")


def extract_phonemes(audio_file_path):
    # Load and process audio
    print(f"Loading audio: {audio_file_path}")
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    speech_array, sr = torchaudio.load(audio_file_path)

    if sr != 16000:
        print("Resampling to 16kHz...")
        resampler = torchaudio.transforms.Resample(sr, 16000)
        speech_array = resampler(speech_array)

    print(f"Original SR: {sr}, channels: {speech_array.shape[0]}")
    audio_duration = len(speech_array.squeeze()) / sr
    print(f"Audio duration: {audio_duration:.2f}s")

    if speech_array.shape[0] > 1:
        speech_array = torch.mean(speech_array, dim=0, keepdim=True)

    speech_array_np = speech_array.squeeze().numpy()
    inputs = phoneme_processor(speech_array.squeeze(), sampling_rate=16000, return_tensors="pt")
    print("Audio processed for model input.\n")

    # -------------------------------
    # Compute global energy threshold for stress
    # -------------------------------
    def compute_rms_energy(audio, start_time, end_time, sample_rate):
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        segment = audio[start_sample:end_sample]
        return np.sqrt(np.mean(segment ** 2)) if len(segment) > 0 else 0.0

    global_rms = compute_rms_energy(speech_array_np, 0, len(speech_array_np) / 16000, 16000)
    energy_threshold = global_rms * ENERGY_THRESHOLD_FACTOR
    print(f"Global RMS energy: {global_rms:.4f}, Stress threshold: {energy_threshold:.4f}\n")

    # -------------------------------
    # Inference
    # -------------------------------
    print("Running inference...")
    with torch.no_grad():
        logits = phoneme_model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)[0].tolist()

    # -------------------------------
    # Frame → phoneme segments with stress
    # -------------------------------
    print("Creating phoneme segments...")
    ipa_segments = []
    ipa_object_segments = []
    gaps = []
    prev_phoneme, start_time = None, 0.0
    special_tokens = ['<pad>', '<s>', '</s>']
    pad_counter = 0
    for i, p_id in enumerate(pred_ids):
        phoneme = id2phoneme.get(p_id, "UNK")
        if phoneme in special_tokens:
            if phoneme == prev_phoneme and phoneme is not None and len(ipa_segments) > 0:
                pad_counter += 1
                if pad_counter > 10 and ipa_segments[-1] != " ":
                    ipa_segments.append(" ")
                    pad_counter = 0
            prev_phoneme = phoneme
            continue
        else:
            pad_counter = 0
        time = i * FRAME_STRIDE
        if phoneme != prev_phoneme and phoneme is not None and phoneme not in special_tokens:
            segment_duration = time - start_time
            segment_rms = compute_rms_energy(speech_array_np, start_time, time, 16000)
            is_stressed = segment_rms > energy_threshold

            if is_stressed:
                ipa_segments.append("ˈ")

            segment_data = {
                "phoneme": ipa_to_phoneme.get(phoneme, phoneme),
                "ipa": phoneme_to_ipa.get(phoneme, phoneme),
                "start": start_time,  # IS NOT ACCURATE
                "end": time,  # IS NOT ACCURATE
                "energy": segment_rms,
                "energyPercentage": segment_rms / global_rms * 100,
                "hasStress": is_stressed
            }

            ipa_object_segments.append(segment_data)
            ipa_segments.append(phoneme)
            start_time = time

        prev_phoneme = phoneme

    print("Output:", "".join(ipa_segments))
    return ipa_segments, ipa_object_segments


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
    audio_ipa_segments: list[dict]
    expected_ipa: str
    audio_ipa: str


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
        actual_ipa_from_text = ipa.convert(result["text"])

        audio_ipa_segments, phones_segments = extract_phonemes(audio_path)
        # Ensure extracted_phonemes is a list of strings, then join
        audio_ipa_string = "".join(audio_ipa_segments)
        # Convert phoneme segments to JSON-serializable format
        ipa_object_segments = [
            {
                "phoneme": se["phoneme"],
                "ipa": se["ipa"],
                "start": float(se["start"]),  # Ensure float for JSON
                "end": float(se["end"]),  # Ensure float for JSON
                "hasStress": bool(se["hasStress"]),
                "energy": float(se["energy"]),  # Convert np.float32 to float
                "energyPercentage": int(se["energyPercentage"])  # Convert np.float32 to float
            }
            for se in phones_segments
        ]
    finally:
        if ext != ".mp3" and os.path.exists(audio_path):
            os.unlink(audio_path)
            logger.debug(f"Deleted temporary WAV file: {audio_path}")
    return {
        "text": result["text"],
        "segments": [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s in result["segments"]
        ],
        "expected_ipa": actual_ipa_from_text,
        "audio_ipa_segments": ipa_object_segments,
        "audio_ipa": audio_ipa_string
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
