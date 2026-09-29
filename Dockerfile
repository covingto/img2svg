FROM python:3.13-slim

WORKDIR /app

# Install git and compilation tools required for custom ops
RUN apt update && apt install -y --no-install-recommends \
    git \
    build-essential \
    ninja-build \ 
    && rm -rf /var/lib/apt/lists/*

# Upgrade packaging tools first to fix wheel path migration
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install remaining dependencies
RUN pip install --no-cache-dir \
    numpy \
    pillow \
    python-multipart \
    fastapi \
    "fastapi[standard]" \
    opencv-python-headless \
    potracer \ 
    cairosvg

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY server.py /app/server.py
COPY index.html /app/index.html

ENV PYTHONUNBUFFERED=1

# CMD ["fastapi", "dev", "server.py"]
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
