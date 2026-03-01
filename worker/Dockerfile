FROM docker.io/pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install YOLOv8 + other ML deps
RUN pip install --no-cache-dir \
    ultralytics==8.3.0 \
    celery[redis]==5.4.0 \
    redis==5.1.1 \
    sqlalchemy==2.0.35 \
    psycopg2-binary==2.9.9 \
    anthropic==0.40.0 \
    httpx==0.27.2 \
    opencv-python-headless==4.10.0.84 \
    ffmpeg-python==0.2.0

COPY . .
