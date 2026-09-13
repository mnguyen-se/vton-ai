# VTON pipeline - GPU image for local use (tested target: RTX 50-series / Blackwell, sm_120).
#
# RTX 5070 needs CUDA >= 12.8 and PyTorch >= 2.7 (that's the first stable
# PyTorch release with native sm_120 kernels). This base image already
# ships a matching torch+cu128 build, so we don't reinstall torch below.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# System libs OpenCV / mediapipe / onnxruntime need that the slim base lacks.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install everything except torch/torchvision (already in the base image).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cache dir for HuggingFace downloads (SD-inpainting, IP-Adapter, CLIP, etc.)
# - mount a volume here so you don't re-download ~6GB every `docker run`.
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface /app/raw_data /app/data /app/lora_output /app/results

CMD ["bash"]
