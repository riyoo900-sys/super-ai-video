# RunPod GitHub deploy — keep in sync with runpod-worker/Dockerfile
FROM runpod/pytorch:2.5.1-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/runpod-volume/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV DIFFUSERS_ATTN_BACKEND=native

RUN apt-get update -qq && apt-get install -y -qq ffmpeg fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt bootstrap.py patch_diffusers.py ./
RUN pip install --no-cache-dir -r requirements.txt \
    && python patch_diffusers.py \
    && python -c "import pathlib,sys; p=pathlib.Path(sys.prefix)/'lib/python3.11/site-packages/diffusers/models/attention_dispatch.py'; t=p.read_text(); assert 'RUNPOD_PATCHED_v7' in t; compile(t,str(p),'exec'); print('patch ok v7')"

COPY wan_engine.py watermark_ffmpeg.py handler.py ./

CMD ["python", "-u", "handler.py"]
