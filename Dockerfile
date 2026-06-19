FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/runpod-volume/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update -qq && apt-get install -y -qq ffmpeg fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

COPY wan_engine.py watermark_ffmpeg.py handler.py ./

CMD ["python", "-u", "handler.py"]
