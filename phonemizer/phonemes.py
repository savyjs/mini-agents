import os
from tkinter import Tk, filedialog
from pydub import AudioSegment
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from phonemizer import phonemize


def select_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an MP4 or Audio file",
        filetypes=[("Audio/Video files", "*.mp4 *.wav *.mp3 *.m4a *.flac")]
    )
    return file_path


def convert_to_wav(input_path):
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".wav":
        return input_path
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    wav_path = "converted_audio.wav"
    audio.export(wav_path, format="wav")
    return wav_path


def transcribe(audio_path):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    waveform, sample_rate = torchaudio.load(audio_path)
    inputs = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt", padding=True)
    logits = model(inputs.input_values).logits
    predicted_ids = logits.argmax(dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription.lower()


def extract_phonemes(text):
    return phonemize(text, language='en-us', backend='espeak', strip=True, with_stress=True, preserve_punctuation=True,
                     preserve_empty_lines=True)


if __name__ == "__main__":
    file_path = select_file()
    if not file_path:
        print("No file selected, exiting.")
        exit()

    wav_path = convert_to_wav(file_path)
    print(f"Processing file: {wav_path}")

    text = transcribe(wav_path)
    print(f"Transcript:\n{text}")

    phonemes = extract_phonemes(text)
    print(f"Phonemes:\n{phonemes}")
