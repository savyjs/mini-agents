import whisper
import subprocess
import os
from tkinter import Tk, filedialog

# 🔹 1. Open file selector GUI to choose .mp4
root = Tk()
root.withdraw()  # Hide the main Tkinter window

mp4_path = filedialog.askopenfilename(
    title="Select an MP4 video file",
    filetypes=[("MP4 files", "*.mp4")]
)

if not mp4_path:
    print("❌ No file selected. Exiting.")
    exit()

print(f"📂 Selected file: {mp4_path}")

# 🔹 2. Extract audio with FFmpeg
wav_path = "extracted_audio.wav"
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

print("🎵 Extracting audio...")
subprocess.run(ffmpeg_command, check=True)
print(f"✅ Audio saved to: {wav_path}")

# 🔹 3. Transcribe using Whisper
# model = whisper.load_model("base")  # or 'small', 'medium', 'large'
model = whisper.load_model("turbo")  # or 'small', 'medium', 'large'
print("🧠 Transcribing...")
result = model.transcribe(wav_path, word_timestamps=True)

# 🔹 4. Display result
print("\n📜 Final Transcription:")
print(result["text"])

print("\n🕓 Segments with Timestamps:")
for segment in result["segments"]:
    print(f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}")
