import os
import math
import json
import wave
import tempfile
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify, send_file, after_this_request
from flask_cors import CORS
from werkzeug.utils import secure_filename
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment

# ---------------------
# Config
# ---------------------
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp3", "wav"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024   # 50 MB (tune as you like)
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "30"))  # 30s chunks
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))       # small for Render free

# Vosk model path: prefer ENV (Dockerfile sets this), else local default
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "vosk-model-small-en-us-0.15")

if not os.path.isdir(VOSK_MODEL_PATH):
    raise FileNotFoundError(
        f"Vosk model not found at '{VOSK_MODEL_PATH}'. "
        f"Download & unzip the model, or set VOSK_MODEL_PATH env var."
    )

# Load model once
model = Model(VOSK_MODEL_PATH)

# Flask
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)  # for GitHub Pages; tighten later if you want


# ---------------------
# Helpers
# ---------------------
def allowed_file(filename: str) -> bool:
    return ('.' in filename) and (filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

def save_upload(file_storage) -> str:
    """Save the uploaded file to a temp path with original extension."""
    ext = os.path.splitext(file_storage.filename)[1].lower() or ".bin"
    tmp = tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, delete=False, suffix=ext)
    file_storage.save(tmp.name)
    return tmp.name

def to_wav_16k_mono(input_path: str) -> str:
    """Convert any supported audio to 16kHz mono WAV using pydub/ffmpeg."""
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    out = tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, delete=False, suffix=".wav")
    audio.export(out.name, format="wav")
    return out.name

def split_wav_into_chunks(wav_path: str, chunk_seconds: int) -> list[str]:
    """Split a WAV file into N small WAV chunk files without loading into RAM."""
    chunks = []
    with wave.open(wav_path, "rb") as wf:
        fr = wf.getframerate()
        sw = wf.getsampwidth()
        ch = wf.getnchannels()
        nframes = wf.getnframes()

        # Sanity: we expect 16kHz mono 16-bit
        if not (fr == 16000 and sw == 2 and ch == 1):
            raise ValueError(f"Unexpected WAV format: fr={fr}, sw={sw}, ch={ch}. Expected 16kHz, 16-bit, mono.")

        total_sec = nframes / fr
        frames_per_chunk = int(chunk_seconds * fr)

        start = 0
        idx = 0
        while start < nframes:
            end = min(start + frames_per_chunk, nframes)
            wf.setpos(start)
            data = wf.readframes(end - start)

            out = tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, delete=False, suffix=f".chunk{idx}.wav")
            with wave.open(out.name, "wb") as ow:
                ow.setnchannels(ch)
                ow.setsampwidth(sw)
                ow.setframerate(fr)
                ow.writeframes(data)

            chunks.append(out.name)
            idx += 1
            start = end
    return chunks

def transcribe_chunk(wav_chunk_path: str) -> str:
    """Transcribe a single 16kHz mono WAV chunk with Vosk."""
    with wave.open(wav_chunk_path, "rb") as wf:
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        while True:
            data = wf.readframes(4000)
            if not data:
                break
            rec.AcceptWaveform(data)
        result = json.loads(rec.FinalResult())
        return result.get("text", "").strip()

def cleanup(paths: list[str]) -> None:
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


# ---------------------
# Routes
# ---------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Invalid file type. Use .mp3 or .wav"}), 400

    tmp_paths = []
    try:
        # 1) Save upload
        src_path = save_upload(f)
        tmp_paths.append(src_path)

        # 2) Ensure WAV 16k mono
        if src_path.lower().endswith(".wav"):
            wav_path = src_path  # trust user-provided WAV; we still validate later
        else:
            wav_path = to_wav_16k_mono(src_path)
            tmp_paths.append(wav_path)

        # 3) Split into small chunks (keeps memory tiny)
        chunks = split_wav_into_chunks(wav_path, CHUNK_SECONDS)
        tmp_paths.extend(chunks)

        # 4) Transcribe in parallel (small pool for 0.1 CPU)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            pieces = list(pool.map(transcribe_chunk, chunks))

        transcript = " ".join([p for p in pieces if p]).strip()

        # 5) Write transcript file
        out_txt = tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, delete=False, suffix=".txt").name
        with open(out_txt, "w", encoding="utf-8") as fp:
            fp.write(transcript or "No speech detected.")
        
        @after_this_request
        def _cleanup(response):
            cleanup(tmp_paths + [out_txt])  # remove everything after response is sent
            return response

        return send_file(out_txt, as_attachment=True, download_name="transcription.txt", mimetype="text/plain")

    except ValueError as ve:
        cleanup(tmp_paths)
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        cleanup(tmp_paths)
        return jsonify({"error": f"Internal error: {e}"}), 500


# ---------------------
# Main
# ---------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
