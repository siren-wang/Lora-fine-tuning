# Docker Setup for LoRA Fine-Tuning Tests

This Docker setup allows you to run the LoRA fine-tuning unit tests in an isolated environment with all dependencies pre-installed.

## Prerequisites

### 1. Install Docker
- **Linux**: Follow [Docker Engine installation](https://docs.docker.com/engine/install/)
- **Windows/Mac**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### 2. Install NVIDIA Docker Support (for GPU)
Required for GPU acceleration:

```bash
# Linux only - install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Verify GPU access:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and run the test
docker-compose up --build

# View logs
docker-compose logs -f

# Clean up
docker-compose down
```

### Option 2: Using Docker Commands

```bash
# Build the image
docker build -t lora-test .

# Run the test with GPU
docker run --gpus all \
  -v $(pwd)/cache:/app/.cache \
  -v $(pwd)/test_lora_output:/app/test_lora_output \
  lora-test

# Run without GPU (CPU only, slower)
docker run \
  -v $(pwd)/cache:/app/.cache \
  -v $(pwd)/test_lora_output:/app/test_lora_output \
  lora-test
```

## What the Test Does

The Docker container will:
1. Download the base model (HuggingFaceTB/SmolLM2-360M-Instruct)
2. Load 10 training samples and 5 validation samples
3. Fine-tune the model for 2 epochs with LoRA
4. Evaluate and compare base vs fine-tuned performance
5. Exit with code 0 if successful

Expected runtime: 5-10 minutes on GPU, 30-60 minutes on CPU.

## Volumes

The setup creates two persistent volumes:

- **`./cache`**: Stores downloaded models and datasets (reused across runs)
- **`./test_lora_output`**: Stores trained model checkpoints

## Configuration

### Use Specific GPU

To use a specific GPU (e.g., GPU 1):

```bash
# Docker Compose: Edit docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=1

# Docker CLI:
docker run --gpus all -e CUDA_VISIBLE_DEVICES=1 lora-test
```

### CPU Only

To force CPU usage (no GPU):

```bash
# Docker CLI:
docker run lora-test

# Docker Compose: Remove the 'deploy' section from docker-compose.yml
```

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker can access GPU
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Restart Docker daemon
sudo systemctl restart docker
```

### Out of Memory

If you get CUDA OOM errors:
- Close other GPU applications
- Reduce batch size in `test_finetuning.py`:
  ```python
  batch_size=1  # Instead of 2
  ```
- Use CPU mode (slower but uses system RAM)

### Slow Downloads

Model downloads can be slow. The first run will take longer as it downloads:
- Base model (~700MB)
- Dataset (~10MB)

Subsequent runs will use cached files.

### Permission Errors

If you get permission errors with volumes:

```bash
# Linux: Fix ownership
sudo chown -R $USER:$USER cache/ test_lora_output/

# Or run with user ID
docker run --user $(id -u):$(id -g) ... lora-test
```

## Interactive Mode

To run commands interactively inside the container:

```bash
# Start interactive shell
docker run --gpus all -it --entrypoint /bin/bash lora-test

# Inside container:
python test_finetuning.py
```

## Cleaning Up

```bash
# Remove containers and volumes
docker-compose down -v

# Remove image
docker rmi lora-test

# Remove all Docker build cache
docker system prune -a
```

## Image Details

- **Base Image**: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
- **Size**: ~8GB
- **CUDA Version**: 12.1
- **PyTorch Version**: 2.1.0

## Running Other Scripts

To run the full training or example scripts:

```bash
# Copy files into container and run
docker run --gpus all -it \
  -v $(pwd):/workspace \
  -w /workspace \
  --entrypoint /bin/bash \
  lora-test

# Inside container:
python example_usage.py
```

## Support

For issues specific to:
- **Docker**: Check [Docker documentation](https://docs.docker.com/)
- **NVIDIA Docker**: Check [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- **The code**: See `README_MODULE.md`
