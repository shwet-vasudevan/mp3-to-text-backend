from flask import Flask, request, jsonify
import os
import wave
import json
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Load Vosk model once at startup
MODEL_PATH = "model"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Vosk model not found in 'model' directory")
model = Model(MODEL_PATH)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CHUNK_LENGTH_MS = 30 * 1000  # 30 seconds
MAX_WORKERS = 4  # safe for 0.1 CPU, keeps RAM < 512MB

def transcribe_chunk(chunk_path):
    wf = wave.open(chunk_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    while True:
        data = wf.readframes(4000)
        if not data:
            break
        rec.AcceptWaveform(data)

    result = json.loads(rec.FinalResult())
    return result.get("text", "")

def process_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    chunks = []
    for i, chunk in enumerate(audio[::CHUNK_LENGTH_MS]):
        chunk_path = os.path.join(UPLOAD_FOLDER, f"chunk_{i}.wav")
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    # Transcribe chunks in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(transcribe_chunk, chunks))

    # Clean up chunk files
    for c in chunks:
        os.remove(c)

    return " ".join(results)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    text = process_audio(file_path)
    os.remove(file_path)

    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
