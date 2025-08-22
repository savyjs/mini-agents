import torch
import torchaudio
import csv
import json
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import os
from pathlib import Path
import sounddevice as sd  # For microphone recording
import numpy as np
import tkinter as tk
from tkinter import filedialog

# -------------------------------
# Config
# -------------------------------
AUDIO_PATH = r"converted_audio.wav"  # Default path for saving recorded or selected audio
MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"
FRAME_STRIDE = 0.02  # seconds per frame
SILENCE_THRESHOLD = 0.1  # Adjusted for finer word segmentation
CSV_EXPORT = "phonemes_output.csv"
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
    "SIL": "‖", "SPN": "…", "UNK": "�"
}


# -------------------------------
# Helper Functions for Audio Input
# -------------------------------
def record_audio(sample_rate=16000):
    """
    Record audio from system microphone with start/stop functionality.
    Press Enter to start and stop recording.
    Returns audio data as numpy array.
    """
    print("Press Enter to start recording...")
    input()  # Wait for user to press Enter
    print("Recording... Press Enter to stop.")
    recording = []

    def callback(indata, frames, time, status):
        if status:
            print(status)
        recording.append(indata.copy())

    stream = sd.InputStream(samplerate=sample_rate, channels=1, callback=callback)
    with stream:
        input()  # Wait for user to press Enter to stop
    audio = np.concatenate(recording, axis=0)
    return audio


def save_audio(audio, sample_rate, output_path):
    """Save numpy array as WAV file."""
    audio_tensor = torch.tensor(audio.T, dtype=torch.float32)  # Convert to torch tensor
    torchaudio.save(output_path, audio_tensor, sample_rate)
    print(f"Audio saved to {output_path}")


def select_audio_file():
    """Open a file dialog to select an audio file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=[("Audio Files", "*.wav *.mp3 *.flac")]
    )
    root.destroy()
    if not file_path:
        raise FileNotFoundError("No file selected.")
    return file_path


# -------------------------------
# Load phoneme vocab from model
# -------------------------------
print("Loading phoneme vocab...")
vocab_file = hf_hub_download(repo_id=MODEL_NAME, filename="vocab.json", cache_dir=CACHE_DIR)
vocab_path = Path(vocab_file)
if not vocab_path.exists():
    raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

with open(vocab_path, "r", encoding="utf-8") as f:
    phoneme_vocab = json.load(f)

# Invert to get ID -> phoneme
id2phoneme = {v: k for k, v in phoneme_vocab.items()}
print(f"Phoneme vocab loaded: {len(id2phoneme)} phonemes.")
print("Phoneme vocabulary:", list(id2phoneme.values()))
print()

# -------------------------------
# Load model + processor
# -------------------------------
print("Loading model and processor...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, do_phonemize=False)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model.eval()
print("Model loaded.\n")

# -------------------------------
# Audio Input Selection
# -------------------------------
print("Select audio input method:")
print("1 - Record from system microphone")
print("2 - Load from audio file")
choice = input("Enter choice (1 or 2): ")

if choice == "1":
    print("Preparing to record audio...")
    audio_data = record_audio(sample_rate=16000)
    save_audio(audio_data, 16000, AUDIO_PATH)
elif choice == "2":
    print("Opening file browser...")
    AUDIO_PATH = select_audio_file()
else:
    raise ValueError("Invalid choice. Please select 1 or 2.")

# -------------------------------
# Load audio
# -------------------------------
print(f"Loading audio: {AUDIO_PATH}")
if not os.path.exists(AUDIO_PATH):
    raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")
speech_array, sr = torchaudio.load(AUDIO_PATH)
print(f"Original SR: {sr}, channels: {speech_array.shape[0]}")

if sr != 16000:
    print("Resampling to 16kHz...")
    resampler = torchaudio.transforms.Resample(sr, 16000)
    speech_array = resampler(speech_array)

# Ensure mono audio
if speech_array.shape[0] > 1:
    speech_array = torch.mean(speech_array, dim=0, keepdim=True)
inputs = processor(speech_array.squeeze(), sampling_rate=16000, return_tensors="pt")
print("Audio processed for model input.\n")

# -------------------------------
# Inference
# -------------------------------
print("Running inference...")
with torch.no_grad():
    logits = model(inputs.input_values).logits

pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
print(f"Total frames: {len(pred_ids)} | Unique phonemes: {len(set(pred_ids))}")
print(f"First 20 predicted IDs: {pred_ids[:20]}\n")

# -------------------------------
# Frame → phoneme segments
# -------------------------------
print("Creating phoneme segments...")
segments = []
prev_id, start_time = None, 0.0

for i, p_id in enumerate(pred_ids):
    phoneme = id2phoneme.get(p_id, "UNK")
    time = i * FRAME_STRIDE
    if phoneme != prev_id:
        if prev_id is not None:
            segments.append({
                "phoneme": prev_id,
                "ipa": phoneme_to_ipa.get(prev_id, prev_id),
                "start": start_time,
                "end": time
            })
        start_time = time
        prev_id = phoneme

if prev_id is not None:
    segments.append({
        "phoneme": prev_id,
        "ipa": phoneme_to_ipa.get(prev_id, prev_id),
        "start": start_time,
        "end": len(pred_ids) * FRAME_STRIDE
    })

print(f"Segments created: {len(segments)}\n")

# -------------------------------
# Group phonemes into word-like chunks
# -------------------------------
print("Grouping phonemes into words...")
words = []
current_word = []

for seg in segments:
    if seg["phoneme"] in ["SIL", "SPN"] or (seg["end"] - seg["start"]) > SILENCE_THRESHOLD:
        if current_word:
            words.append(current_word)
            current_word = []
    else:
        current_word.append(seg)
if current_word:
    words.append(current_word)

print(f"Total words detected: {len(words)}\n")

# -------------------------------
# Print results
# -------------------------------
print("=== Phoneme Segments ===")
for s in segments:
    print(f"{s['start']:.2f}-{s['end']:.2f} : {s['phoneme']} -> {s['ipa']}")

print("\n=== Word-like groups ===")
for i, word in enumerate(words, 1):
    ipa_word = "-".join([seg["ipa"] for seg in word])
    print(f"Word {i}: {ipa_word}")

print("\n=== Detailed Word Groups ===")
for i, word in enumerate(words, 1):
    ipa_word = "-".join([seg["ipa"] for seg in word])
    phonemes = [seg["phoneme"] for seg in word]
    duration = sum(seg["end"] - seg["start"] for seg in word)
    print(f"Word {i}: IPA={ipa_word}, Phonemes={phonemes}, Duration={duration:.2f}s")

# -------------------------------
# Export CSV
# -------------------------------
print(f"\nExporting to CSV: {CSV_EXPORT} ...")
with open(CSV_EXPORT, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["word_index", "start", "end", "phoneme", "IPA"])
    for idx, word in enumerate(words, 1):
        for seg in word:
            writer.writerow([idx, f"{seg['start']:.3f}", f"{seg['end']:.3f}", seg["phoneme"], seg["ipa"]])

print("Export complete.\nAll done!")