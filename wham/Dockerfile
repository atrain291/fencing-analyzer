FROM docker.io/pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    libpq-dev \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps — carefully ordered to avoid conflicts
# Core WHAM dependencies (skip mmcv — we vendor the needed ops)
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    smplx \
    einops \
    yacs \
    loguru \
    tqdm \
    scikit-image \
    opencv-python-headless \
    imageio[ffmpeg] \
    timm==0.6.13 \
    progress \
    joblib \
    chumpy \
    gdown \
    celery[redis]==5.4.0 \
    redis==5.1.1 \
    sqlalchemy==2.0.35 \
    psycopg2-binary==2.9.9

# Clone WHAM (without recursive — we skip ViTPose/DPVO submodules)
RUN git clone --depth 1 https://github.com/yohanshin/WHAM.git /opt/wham

# Download model checkpoints (WHAM + HMR2)
# Note: SMPL body models must be downloaded separately after accepting the license
# at https://smpl.is.tue.mpg.de/ and placed in /opt/wham/dataset/body_models/
RUN mkdir -p /opt/wham/checkpoints && \
    gdown "1i7kt9N2PEatMP66AEbBxfuJkPGA1tcef" -O /opt/wham/checkpoints/wham_vit_bedlam_w_3dpw.pth.tar || true && \
    gdown "1mQSP4T0xN8snqDqhWb4fCbSEHUKJB7_G" -O /opt/wham/checkpoints/hmr2a.ckpt || true

ENV PYTHONPATH="/opt/wham:${PYTHONPATH}"

COPY . .
