FROM python:3.11-slim

RUN apt-get update &&     apt-get install -y --no-install-recommends ffmpeg curl unzip &&     rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the Vosk small English model
RUN mkdir -p /models &&     curl -L -o /tmp/vosk.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip &&     unzip /tmp/vosk.zip -d /models &&     rm /tmp/vosk.zip

COPY . .

ENV PYTHONUNBUFFERED=1
ENV VOSK_MODEL_PATH=/models/vosk-model-small-en-us-0.15

EXPOSE 8000

# Render provides ; default to 8000 locally
CMD exec gunicorn app:app --bind 0.0.0.0:8000 --workers 2 --threads 4 --timeout 180

