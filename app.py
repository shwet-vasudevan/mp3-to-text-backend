import os
import wave
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from vosk import Model, KaldiRecognizer
import subprocess
from pydub import AudioSegment

app = Flask(__name__)

# Config
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Vosk model
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "vosk-model-small-en-us-0.15")
if not os.path.exists(VOSK_MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {VOSK_MODEL_PATH}")
model = Model(VOSK_MODEL_PATH)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

def transcribe_chunk(wav_chunk_path):
    wf = wave.open(wav_chunk_path, "rb")
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
    return " ".join([res.get("text", "") for res in results])

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Convert to WAV (mono, 16kHz)
    wav_path = os.path.splitext(filepath)[0] + ".wav"
    subprocess.run(
        ["ffmpeg", "-i", filepath, "-ar", "16000", "-ac", "1", wav_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Split into smaller chunks (1 minute each)
    audio = AudioSegment.from_wav(wav_path)
    chunk_length_ms = 60 * 1000  # 1 minute
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

    full_text = ""
    for i, chunk in enumerate(chunks):
        chunk_filename = f"{os.path.splitext(filename)[0]}_chunk{i}.wav"
        chunk_path = os.path.join(UPLOAD_FOLDER, chunk_filename)
        chunk.export(chunk_path, format="wav")
        full_text += " " + transcribe_chunk(chunk_path)

    # Save final text file
    text_filename = os.path.splitext(filename)[0] + ".txt"
    text_path = os.path.join(UPLOAD_FOLDER, text_filename)
    with open(text_path, "w") as f:
        f.write(full_text.strip())

    return jsonify({"text": full_text.strip(), "text_file": text_filename})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
