FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Australia/Melbourne

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY container/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gdown

# Copy air-400 source code
COPY main.py infer.py /app/
COPY dataloaders /app/dataloaders/
COPY models /app/models/
COPY processors /app/processors/
COPY trainers /app/trainers/
COPY loss /app/loss/

# Build pyflow
RUN git clone https://github.com/pathak22/pyflow.git && \
    cd pyflow && \
    python setup.py build_ext -i && \
    cp pyflow.cpython-*.so /app/ && \
    cd .. && rm -rf pyflow

# Copy service files
COPY container/service.py /app/
COPY container/config.yaml /app/
COPY container/entrypoint.sh /app/

RUN chmod +x /app/entrypoint.sh

# Create directories
RUN mkdir -p /app/models_pretrained/checkpoint /app/models_pretrained/detectors /app/output

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "service.py"]
