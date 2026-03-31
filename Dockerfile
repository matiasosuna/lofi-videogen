FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

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