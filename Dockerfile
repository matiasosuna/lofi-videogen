FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Install all ML + server dependencies in one layer
RUN pip install --no-cache-dir \
    diffusers \
    transformers \
    accelerate \
    sentencepiece \
    protobuf \
    Pillow \
    fastapi \
    uvicorn \
    python-multipart \
    safetensors

# Copy server
COPY server.py /opt/server.py

EXPOSE 8000

WORKDIR /opt
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]