# ──────────────────────────────────────────────────────────────────────────────
# Dockerfile for an MCP server that performs CUDA-accelerated inference
# Serves HTTP traffic on port 8000, code rooted at /mcp
# ──────────────────────────────────────────────────────────────────────────────

############################
# 1. BASE – CUDA + Ubuntu  #
############################
FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04 AS base
LABEL maintainer="ryan.sherby@columbia.edu"

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1  \
    PYTHONUNBUFFERED=1          \
    # Make sure torch/cuDNN find the right GPUs when multiple are present
    CUDA_VISIBLE_DEVICES=0

# Install Python and common system deps in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev build-essential \
        git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*


# Install download tools
RUN apt-get update && apt-get install -y --no-install-recommends \
      git-lfs \
  && rm -rf /var/lib/apt/lists/* \
  && git lfs install
    

############################
# 2. BUILD – install deps  #
############################
FROM base AS builder

# Optionally enable faster, parallel pip resolution
RUN pip3 install --no-cache-dir pip==24.0 setuptools wheel

# Copy only requirements first (leverages Docker layer caching)
WORKDIR /tmp/reqs
COPY requirements.txt ./
COPY pyproject.toml ./


RUN if [ -f "requirements.txt" ]; then \
        pip3 wheel --wheel-dir /wheels -r requirements.txt ; \
    fi

# Install Flit build backend for local wheel from pyproject.toml
RUN pip3 install --no-cache-dir \
        pip==24.0 \
        build \
        flit-core

# Build the local package into a wheel
WORKDIR /src
COPY pyproject.toml ./
COPY viper/src ./viper/src
COPY viper/__init__.py ./viper/__init__.py
COPY viper/mcp ./viper/mcp
COPY viper/configs ./viper/configs
RUN python3 -m build --wheel --no-isolation --outdir /wheels




############################
# 3. FINAL – runtime image #
############################
FROM base AS runtime

WORKDIR /app


# Copy Python wheels + install (smaller & faster than pip install from PyPI)
COPY --from=builder /wheels /wheels
RUN pip3 install --no-cache-dir /wheels/* && rm -rf /wheels

RUN pip3 install --no-cache-dir huggingface_hub gdown

# Copy the remainder of the project source
COPY viper /app/viper
COPY download-models.sh /app/download-models.sh


RUN useradd --create-home appuser

# Create a cache directory that is owned by the app user
ENV VIPER_CACHE_DIR=/var/cache/viper
RUN mkdir -p /var/cache/viper && chown -R appuser:appuser /var/cache/viper

# Make sure model dirs are writable
ENV VIPER_MODELS_DIR=/app/viper/pretrained_models

RUN mkdir -p /app/viper/pretrained_models && chown -R appuser:appuser /app/viper

# Download pretrained models
RUN /app/download-models.sh && chown -R appuser:appuser /app/viper/pretrained_models && \
    rm -rf /app/download-models.sh

# Clear Hugging Face cache to save space
RUN rm -rf ./.cache/huggingface

USER appuser


# Leverage NVIDIA environment variables; defaults are usually fine
# but uncomment to pin a specific GPU architecture, e.g.:
# ENV TORCH_CUDA_ARCH_LIST="8.6+PTX"

EXPOSE 8000


CMD ["fastmcp", "run", \
    "/app/viper/mcp/server.py:mcp", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--transport", "streamable-http"]
