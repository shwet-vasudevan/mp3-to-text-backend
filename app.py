import os
import wave
import json
from flask import Flask, request, send_file, jsonify
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
from werkzeug.utils import secure_filename
from flask_cors import CORS

# --- Config ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB upload max
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "vosk-model-small-en-us-0.15")

# --- Init Flask ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://<your-username>.github.io"]}})  # replace <your-username>
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return isinstance(filename, str) and filename.lower().endswith('.mp3')

def mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_file(mp3_path, format="mp3")
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(wav_path, format="wav")

def audio_to_text(wav_file_path):
    if not os.path.isdir(VOSK_MODEL_PATH):
        raise FileNotFoundError(f"Vosk model not found at '{VOSK_MODEL_PATH}'")
    model = Model(VOSK_MODEL_PATH)
    wf = wave.open(wav_file_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    parts = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            out = json.loads(rec.Result())
            t = out.get("text", "")
            if t:
                parts.append(t)

    final = json.loads(rec.FinalResult())
    ft = final.get("text", "")
    if ft:
        parts.append(ft)

    wf.close()
    transcript = " ".join(parts).strip()
    return transcript if transcript else "No speech detected."

def cleanup_files(paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except:
            pass

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Only .mp3 allowed."}), 400

    original_filename = secure_filename(file.filename)
    base = os.path.splitext(original_filename)[0]
    temp_mp3 = temp_wav = txt_path = None

    try:
        temp_mp3 = os.path.join(UPLOAD_FOLDER, f"{base}_{os.urandom(6).hex()}.mp3")
        file.save(temp_mp3)
        temp_wav = os.path.join(UPLOAD_FOLDER, f"{base}_{os.urandom(6).hex()}.wav")
        mp3_to_wav(temp_mp3, temp_wav)
        transcript = audio_to_text(temp_wav)
        txt_path = os.path.join(UPLOAD_FOLDER, f"{base}_{os.urandom(6).hex()}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        cleanup_files([temp_mp3, temp_wav])
        return send_file(txt_path, as_attachment=True, download_name=f"{base}.txt", mimetype="text/plain")
    except Exception as e:
        cleanup_files([temp_mp3, temp_wav])
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
