import torch
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf

# Load a pretrained Wav2Vec2 model fine-tuned for phoneme recognition
# This one is trained with eSpeak phoneme set (close to IPA)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")

# Load audio
speech_array, sampling_rate = sf.read("converted_audio.wav")

# If stereo, take first channel
if len(speech_array.shape) > 1:
    speech_array = speech_array[:,0]

# Preprocess
inputs = processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

# Run model
with torch.no_grad():
    logits = model(**inputs).logits

# Decode to phonemes
pred_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(pred_ids)[0]

print("IPA-like phoneme transcription:")
print(transcription)
