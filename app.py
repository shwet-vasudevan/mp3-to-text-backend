import os
import wave
import json
from flask import Flask, render_template, request, send_file, jsonify
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
from werkzeug.utils import secure_filename
from flask_cors import CORS

# --- Config ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3'}
MAX_CONTENT_LENGTH = 300 * 1024 * 1024  # 300 MB
# Allow Render to set a model path via ENV; fallback to common default
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "vosk-model-small-en-us-0.15")

# --- Init Flask ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # later, restrict to your Pages domain
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename: str) -> bool:
    return isinstance(filename, str) and filename.lower().endswith('.mp3')

def mp3_to_wav(mp3_path: str, wav_path: str):
    audio = AudioSegment.from_file(mp3_path, format="mp3")
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(wav_path, format="wav")

def audio_to_text(wav_file_path: str) -> str:
    if not os.path.isdir(VOSK_MODEL_PATH):
        raise FileNotFoundError(f"Vosk model not found at '{VOSK_MODEL_PATH}'.")
    model = Model(VOSK_MODEL_PATH)
    wf = wave.open(wav_file_path, "rb")

    channels = wf.getnchannels()
    sampwidth = wf.getsampwidth()
    framerate = wf.getframerate()
    if channels != 1 or sampwidth != 2 or framerate != 16000:
        wf.close()
        raise ValueError(f"Invalid WAV format: channels={channels}, sample_width={sampwidth}, framerate={framerate}. "
                         "Expected mono, 16-bit, 16kHz.")

    rec = KaldiRecognizer(model, framerate)
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
        except Exception as e:
            app.logger.warning(f"Cleanup failed for {p}: {e}")

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    app.logger.info("Upload hit")
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
        temp_mp3 = os.path.join(app.config['UPLOAD_FOLDER'], f"{base}_{os.urandom(6).hex()}.mp3")
        file.save(temp_mp3)

        temp_wav = os.path.join(app.config['UPLOAD_FOLDER'], f"{base}_{os.urandom(6).hex()}.wav")
        mp3_to_wav(temp_mp3, temp_wav)

        transcript = audio_to_text(temp_wav)

        txt_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base}_{os.urandom(6).hex()}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        cleanup_files([temp_mp3, temp_wav])
        return send_file(txt_path, as_attachment=True, download_name=f"{base}.txt", mimetype='text/plain')

    except FileNotFoundError as e:
        cleanup_files([temp_mp3, temp_wav])
        return jsonify({"error": str(e)}), 500
    except ValueError as e:
        cleanup_files([temp_mp3, temp_wav])
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.exception("Unhandled error")
        cleanup_files([temp_mp3, temp_wav])
        return jsonify({"error": "Internal server error: " + str(e)}), 500

if __name__ == '__main__':
    # local dev only
    app.run(host='0.0.0.0', port=5000, debug=True)

