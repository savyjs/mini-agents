import json
import wave
from vosk import Model, KaldiRecognizer

# Hardcoded paths (replace with your own if needed)
AUDIO_PATH = "converted_audio.wav"
MODEL_PATH = r"E:/LLMs/projects/vosk/vosk-model-small-en-us-0.15"
MAP_PATH   = "phone2ipa.tsv"


def load_phone_map(path):
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            phone, ipa = line.strip().split("\t")
            mapping[phone] = ipa
    return mapping


def read_wav(path):
    wf = wave.open(path, "rb")
    if wf.getnchannels() != 1:
        raise ValueError("Please use mono WAV (1 channel).")
    if wf.getsampwidth() != 2:
        raise ValueError("Expect 16-bit PCM WAV.")
    return wf


def main():
    phone_map = load_phone_map(MAP_PATH)
    model = Model(MODEL_PATH)
    wf = read_wav(AUDIO_PATH)

    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result()))
    results.append(json.loads(rec.FinalResult()))

    # collect phones if available
    all_phones = []
    for r in results:
        if "phones" in r:
            for p in r["phones"]:
                start = p.get("start", 0)
                end = p.get("end", 0)
                phone = p.get("phone")
                ipa = phone_map.get(phone, phone)
                all_phones.append((start, end, phone, ipa))

    if not all_phones:
        print("⚠ No phone info in this model’s output.")
        return

    print("start,end,kaldi_phone,ipa")
    for start, end, phone, ipa in all_phones:
        print(f"{start:.3f},{end:.3f},{phone},{ipa}")


if __name__ == "__main__":
    main()
