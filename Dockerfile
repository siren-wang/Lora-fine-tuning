# Dockerfile for LoRA Fine-Tuning Unit Tests
# Lightweight image for running test_finetuning.py

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    transformers \
    datasets \
    peft \
    accelerate \
    bitsandbytes \
    trl \
    torch \
    numpy

# Copy only necessary files
COPY lora_finetuning.py /app/
COPY test_finetuning.py /app/

# Create cache and output directories
RUN mkdir -p /app/.cache/huggingface && \
    mkdir -p /app/test_lora_output

# Make test script executable
RUN chmod +x /app/test_finetuning.py

# Default command: run the test
CMD ["python", "test_finetuning.py"]
