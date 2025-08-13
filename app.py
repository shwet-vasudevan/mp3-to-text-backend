import os
import wave
import json
import math
import tempfile
from flask import Flask, request, jsonify, send_file
from vosk import Model, KaldiRecognizer
from concurrent.futures import ThreadPoolExecutor

# ---------------------
# CONFIG
# ---------------------
# Allow override from Render environment variable, default to small English model
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "vosk-model-small-en-us-0.15")

if not os.path.exists(VOSK_MODEL_PATH):
    raise FileNotFoundError(
        f"❌ Vosk model not found at '{VOSK_MODEL_PATH}'. "
        f"Download and extract it from https://alphacephei.com/vosk/models "
        f"or set the VOSK_MODEL_PATH environment variable."
    )

# Load model once at startup
print(f"✅ Loading Vosk model from {VOSK_MODEL_PATH} ...")
model = Model(VOSK_MODEL_PATH)
print("✅ Model loaded successfully.")

# Flask app
app = Flask(__name__)

# ---------------------
# HELPERS
# ---------------------
def transcribe_chunk(chunk_path):
    """Transcribe a single WAV chunk."""
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

def split_audio(input_path, chunk_length_s=30):
    """Split audio into smaller chunks to fit memory limits."""
    wf = wave.open(input_path, "rb")
    framerate = wf.getframerate()
    nframes = wf.getnframes()
    duration_s = nframes / framerate

    chunks = []
    for i in range(0, math.ceil(duration_s / chunk_length_s)):
        start_frame = int(i * chunk_length_s * framerate)
        end_frame = min(int((i + 1) * chunk_length_s * framerate), nframes)

        wf.setpos(start_frame)
        frames = wf.readframes(end_frame - start_frame)

        chunk_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(chunk_file.name, "wb") as out_wav:
            out_wav.setnchannels(wf.getnchannels())
            out_wav.setsampwidth(wf.getsampwidth())
            out_wav.setframerate(framerate)
            out_wav.writeframes(frames)

        chunks.append(chunk_file.name)

    return chunks

# ---------------------
# ROUTES
# ---------------------
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save to temp file
    tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    file.save(tmp_input.name)

    # Split audio into chunks
    chunks = split_audio(tmp_input.name, chunk_length_s=30)

    # Transcribe chunks in parallel (limit threads for Render CPU constraints)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = executor.map(transcribe_chunk, chunks)

    # Combine results
    full_text = " ".join(results)

    # Save final transcription to file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
    with open(output_path, "w") as f:
        f.write(full_text)

    return send_file(output_path, as_attachment=True, download_name="transcription.txt")

# ---------------------
# ENTRY POINT
# ---------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
