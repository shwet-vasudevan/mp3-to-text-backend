import os
import wave
import json
from flask import Flask, request, send_file, jsonify
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
from werkzeug.utils import secure_filename
from flask_cors import CORS

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # limit to 100 MB
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "vosk-model-small-en-us-0.15")
CHUNK_DURATION_MS = 30_000  # 30-second chunks

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename: str) -> bool:
    return filename.lower().endswith('.mp3')

def mp3_to_wav_chunks(mp3_path: str):
    audio = AudioSegment.from_file(mp3_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    chunks = []
    for i, start_ms in enumerate(range(0, len(audio), CHUNK_DURATION_MS)):
        chunk = audio[start_ms:start_ms + CHUNK_DURATION_MS]
        chunk_path = os.path.join(UPLOAD_FOLDER, f"chunk_{i}.wav")
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

def transcribe_chunk(wav_path: str, model: Model) -> str:
    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    parts = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            out = json.loads(rec.Result())
            if out.get("text"):
                parts.append(out["text"])
    final = json.loads(rec.FinalResult())
    if final.get("text"):
        parts.append(final["text"])
    wf.close()
    return " ".join(parts).strip()

def cleanup_files(paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except:
            pass

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    original_filename = secure_filename(file.filename)
    base = os.path.splitext(original_filename)[0]
    temp_mp3 = os.path.join(UPLOAD_FOLDER, f"{base}_{os.urandom(6).hex()}.mp3")
    file.save(temp_mp3)

    try:
        # Load Vosk model once
        if not os.path.isdir(VOSK_MODEL_PATH):
            raise FileNotFoundError(f"Vosk model not found at '{VOSK_MODEL_PATH}'")
        model = Model(VOSK_MODEL_PATH)

        # Split MP3 into WAV chunks and transcribe
        chunks = mp3_to_wav_chunks(temp_mp3)
        transcript_parts = [transcribe_chunk(chunk, model) for chunk in chunks]

        transcript = "\n".join(transcript_parts).strip()
        txt_path = os.path.join(UPLOAD_FOLDER, f"{base}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        cleanup_files([temp_mp3] + chunks)
        return send_file(txt_path, as_attachment=True, download_name=f"{base}.txt", mimetype='text/plain')
    except Exception as e:
        cleanup_files([temp_mp3])
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
