import torch
import torchaudio
import csv
import json
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import os
from pathlib import Path
import sounddevice as sd
import numpy as np
import tkinter as tk
from tkinter import filedialog

# -------------------------------
# Config
# -------------------------------
AUDIO_PATH = r"converted_audio.wav"
MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"
FRAME_STRIDE = 0.02  # seconds per frame
SILENCE_THRESHOLD = 0.2  # Silence duration for word separation
MIN_WORD_DURATION = 0.1  # Minimum duration for a word
MIN_GAP_DURATION = 0.05  # Reduced for better gap detection
ENERGY_THRESHOLD_FACTOR = 1.5  # Factor for stress detection
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
    "AEː": "æː", "AHː": "ɐː", "IYː": "iː", "UWː": "uː", "ERː": "ɚ",
    "OI": "ɔɪ", "OU": "oʊ", "AI": "aɪ", "AU": "aʊ",
    "TS": "ts", "Tʃ": "tʃ", "Dʒ": "dʒ", "ɡʲ": "ɡʲ",
    "NY": "ɲ", "LH": "ɬ", "LL": "ʎ", "RH": "ʁ",
    "SIL": "<s>", "PAD": "<pad>", "EOS": "</s>", "UNK": "<unk>"
}

# Create the inverse mapping
ipa_to_phoneme = {ipa: phoneme for phoneme, ipa in phoneme_to_ipa.items()}


# -------------------------------
# Helper Functions
# -------------------------------
def record_audio(sample_rate=16000):
    print("Press Enter to start recording...")
    input()
    print("Recording... Press Enter to stop.")
    recording = []

    def callback(indata, frames, time, status):
        if status:
            print(status)
        recording.append(indata.copy())

    stream = sd.InputStream(samplerate=sample_rate, channels=1, callback=callback)
    with stream:
        input()
    audio = np.concatenate(recording, axis=0)
    return audio


def save_audio(audio, sample_rate, output_path):
    audio_tensor = torch.tensor(audio.T, dtype=torch.float32)
    torchaudio.save(output_path, audio_tensor, sample_rate)
    print(f"Audio saved to {output_path}")


def select_audio_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=[("Audio Files", "*.wav *.mp3 *.flac")]
    )
    root.destroy()
    if not file_path:
        raise FileNotFoundError("No file selected.")
    return file_path


def compute_rms_energy(audio, start_time, end_time, sample_rate):
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    segment = audio[start_sample:end_sample]
    if len(segment) == 0:
        return 0.0
    return np.sqrt(np.mean(segment ** 2))


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

id2phoneme = {v: k for k, v in phoneme_vocab.items()}
print(f"Phoneme vocab loaded: {len(id2phoneme)} phonemes.")
# print("id2phoneme:", id2phoneme)

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
audio_duration = len(speech_array.squeeze()) / sr
print(f"Audio duration: {audio_duration:.2f}s")

if sr != 16000:
    print("Resampling to 16kHz...")
    resampler = torchaudio.transforms.Resample(sr, 16000)
    speech_array = resampler(speech_array)

if speech_array.shape[0] > 1:
    speech_array = torch.mean(speech_array, dim=0, keepdim=True)
speech_array_np = speech_array.squeeze().numpy()
inputs = processor(speech_array.squeeze(), sampling_rate=16000, return_tensors="pt")
print("Audio processed for model input.\n")

# -------------------------------
# Compute global energy threshold for stress
# -------------------------------
global_rms = compute_rms_energy(speech_array_np, 0, len(speech_array_np) / 16000, 16000)
energy_threshold = global_rms * ENERGY_THRESHOLD_FACTOR
print(f"Global RMS energy: {global_rms:.4f}, Stress threshold: {energy_threshold:.4f}\n")

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
# Frame → phoneme segments with stress
# -------------------------------
print("Creating phoneme segments...")
pure_segments = []
segments = []
gaps = []
prev_phoneme, start_time = None, 0.0
special_tokens = ['<pad>', '<s>', '</s>']
pad_counter = 0
for i, p_id in enumerate(pred_ids):
    phoneme = id2phoneme.get(p_id, "UNK")
    print("phoneme:", phoneme)
    if phoneme in special_tokens:
        if phoneme == prev_phoneme and phoneme is not None and len(pure_segments) > 0:
            print("Skipped special token:", phoneme)
            pad_counter = 1 + pad_counter
            if pad_counter > 5 and pure_segments[-1] != " ":
                pure_segments.append(" ")
                pad_counter = 0
        prev_phoneme = phoneme
        continue
    time = i * FRAME_STRIDE
    if phoneme != prev_phoneme and phoneme is not None and phoneme not in special_tokens:
        print("detected phoneme:", phoneme)
        segment_duration = time - start_time
        segment_rms = compute_rms_energy(speech_array_np, start_time, time, 16000)
        is_stressed = "stressed" if segment_rms > energy_threshold else "unstressed"
        segment_data = {
            "phoneme": ipa_to_phoneme.get(phoneme, phoneme),
            "ipa": phoneme_to_ipa.get(phoneme, phoneme),
            "start": start_time,  # IS NOT ACCURATE
            "end": time,  # IS NOT ACCURATE
            "energy": segment_rms,
            "energyPercentage": segment_rms / global_rms * 100,
            "stress": is_stressed
        }

        print("added phoneme:", phoneme)
        segments.append(segment_data)
        pure_segments.append(phoneme)
        start_time = time
    else:
        print("Invalid or Duplicated: ", phoneme)
    prev_phoneme = phoneme

print(f"Segments created: {len(segments)}")
print("Phoneme segments:", [(s['phoneme'], s['ipa'], s['start'], s['end'], s['stress']) for s in segments])
print(f"Gaps detected: {len(gaps)}\n")

# -------------------------------
# Print results in IPA format
# -------------------------------
print("=== Phoneme Segments ===")
for s in segments:
    print(
        f"{s['start']:.2f}-{s['end']:.2f} : {s['phoneme']} -> {s['ipa']} ({s['stress']}, energy={s['energyPercentage']:.1f}%)")

# Save JSON to file and wait until fully written
# Convert floats to regular Python floats
safe_segments = []
for seg in segments:
    safe_segments.append({
        "start": float(seg["start"]),
        "end": float(seg["end"]),
        "IPA": seg["ipa"],
        "Phoneme": seg["phoneme"],
        "hasStress": seg["stress"] == "stressed",
        "energy": float(seg["energy"]),
        "energyPercentage": float(seg["energyPercentage"])
    })

# Save JSON safely
JSON_EXPORT = "phoneme_segments.json"
with open(JSON_EXPORT, "w", encoding="utf-8") as f:
    json.dump(safe_segments, f, ensure_ascii=False, indent=4)
    f.flush()
    os.fsync(f.fileno())

print("Output:", "".join(pure_segments))
print("Export complete.\nAll done!")
