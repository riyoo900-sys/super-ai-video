# RunPod GitHub deploy — based on build 0cd087e (successful) + diffusers patch
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Use container disk (/tmp), not /runpod-volume — network volume is often too small for ~10GB model
ENV HF_HOME=/tmp/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV DIFFUSERS_ATTN_BACKEND=native
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1

RUN apt-get update -qq && apt-get install -y -qq ffmpeg fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt bootstrap.py patch_diffusers.py ./
RUN pip install --no-cache-dir -r requirements.txt \
    && python patch_diffusers.py

COPY wan_engine.py watermark_ffmpeg.py handler.py start.py ./

CMD ["python", "-u", "start.py"]
