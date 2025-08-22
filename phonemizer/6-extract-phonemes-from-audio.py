import torch
import torchaudio
import json
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from huggingface_hub import hf_hub_download
from pathlib import Path
import numpy as np
import os

# -------------------------------
# Config
# -------------------------------
MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"
FRAME_STRIDE = 0.02  # seconds per frame
ENERGY_THRESHOLD_FACTOR = 1.5  # Factor for stress detection
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
ipa_to_phoneme = {ipa: phoneme for phoneme, ipa in phoneme_to_ipa.items()}

# -------------------------------
# Helper Functions
# -------------------------------
def compute_rms_energy(audio, start_time, end_time, sample_rate):
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    segment = audio[start_sample:end_sample]
    if len(segment) == 0:
        return 0.0
    return np.sqrt(np.mean(segment ** 2))


# -------------------------------
# Main Function
# -------------------------------
def extract_pure_phoneme_segments(audio_path: str):
    # Load audio
    speech_array, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        speech_array = resampler(speech_array)
    if speech_array.shape[0] > 1:
        speech_array = torch.mean(speech_array, dim=0, keepdim=True)
    speech_array_np = speech_array.squeeze().numpy()

    # Load phoneme vocab
    vocab_file = hf_hub_download(repo_id=MODEL_NAME, filename="vocab.json", cache_dir=CACHE_DIR)
    with open(vocab_file, "r", encoding="utf-8") as f:
        phoneme_vocab = json.load(f)
    id2phoneme = {v: k for k, v in phoneme_vocab.items()}

    # Load model + processor
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, do_phonemize=False)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    model.eval()

    # Process input
    inputs = processor(speech_array.squeeze(), sampling_rate=16000, return_tensors="pt")

    # Compute global energy threshold
    global_rms = compute_rms_energy(speech_array_np, 0, len(speech_array_np)/16000, 16000)
    energy_threshold = global_rms * ENERGY_THRESHOLD_FACTOR

    # Inference
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)[0].tolist()

    # Create phoneme segments
    pure_segments = []
    prev_phoneme, start_time = None, 0.0
    special_tokens = ['<pad>', '<s>', '</s>']
    pad_counter = 0

    for i, p_id in enumerate(pred_ids):
        phoneme = id2phoneme.get(p_id, "UNK")
        if phoneme in special_tokens:
            if phoneme == prev_phoneme and phoneme is not None and len(pure_segments) > 0:
                pad_counter += 1
                if pad_counter > 5 and pure_segments[-1] != " ":
                    pure_segments.append(" ")
                    pad_counter = 0
            prev_phoneme = phoneme
            continue

        time = i * FRAME_STRIDE
        if phoneme != prev_phoneme and phoneme not in special_tokens:
            segment_rms = compute_rms_energy(speech_array_np, start_time, time, 16000)
            is_stressed = "stressed" if segment_rms > energy_threshold else "unstressed"
            pure_segments.append(phoneme)
            start_time = time
        prev_phoneme = phoneme

    return pure_segments

